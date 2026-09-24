#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1 training: LinearSVC.

Outer GroupKFold evaluation with one group-aware inner train/validation
holdout per outer fold. Hyperparameters are selected on inner validation AUC.
The supplied 0/1 labels and existing feature values are used without relabeling
or additional standardization. The CV evaluation model is fitted on inner
training samples; the separately saved full-data model is not a CV estimator.

Run from the repository root:
    python scripts/training/task1/train_linear_svm.py --help

Default experiment settings: configs/task1/linear_svm.json.
The numerical training protocol is retained from the supplied research script.
"""

import json
import os
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import LinearSVC

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 配置：关键评估设置与 train_xgboost.py 对齐
# =============================================================================
from _runtime import load_default_config, public_path, run_cli

DEFAULT_CONFIG, REPO_ROOT = load_default_config(__file__, "linear_svm")
RANDOM_SEED = DEFAULT_CONFIG["parameters"]["RANDOM_SEED"]
N_FOLDS = DEFAULT_CONFIG["parameters"]["N_FOLDS"]
N_TRIALS = DEFAULT_CONFIG["parameters"]["N_TRIALS"]
INNER_N_SPLITS = DEFAULT_CONFIG["parameters"]["INNER_N_SPLITS"]
INNER_VALID_SIZE = DEFAULT_CONFIG["parameters"]["INNER_VALID_SIZE"]
INNER_MAX_RETRIES = DEFAULT_CONFIG["parameters"]["INNER_MAX_RETRIES"]
DROP_NEAR_CONSTANT = DEFAULT_CONFIG["parameters"]["DROP_NEAR_CONSTANT"]
NEAR_CONST_STD_TOL = DEFAULT_CONFIG["parameters"]["NEAR_CONST_STD_TOL"]
N_JOBS = DEFAULT_CONFIG["runtime"]["n_jobs"]
DATA_PATH = str(REPO_ROOT / DEFAULT_CONFIG["data"]["path"])
DATA_REFERENCE = public_path(DATA_PATH, REPO_ROOT)
OUTDIR = str(REPO_ROOT / DEFAULT_CONFIG["output"]["root"])
LABEL_MAP = {0: 0, 1: 1}
INVERSE_LABEL_MAP = {0: 0, 1: 1}
FIXED_MODEL_PARAMS = DEFAULT_CONFIG["fixed_model_params"]
SEARCH_SPACE = DEFAULT_CONFIG["search_space"]
BASELINE_PARAMS = DEFAULT_CONFIG["baseline_params"]


def to_builtin(value):
    """把 numpy 标量转换为 JSON/joblib 友好的 Python 标量。"""
    return value.item() if isinstance(value, np.generic) else value


def load_data(data_path):
    """读取数据并执行与线性支持向量机运行直接相关的必要检查。"""
    df = pd.read_csv(data_path)
    if df.shape[1] < 3:
        raise ValueError("[数据] CSV 至少需要 protein_id、1 个特征和标签共 3 列。")

    X = df.iloc[:, 1:-1].copy()
    y_raw = df.iloc[:, -1].astype(int).copy()
    groups = df.iloc[:, 0].copy()
    y = y_raw.map(LABEL_MAP)

    if groups.isna().any():
        raise ValueError("[数据] protein_id 中存在 NaN，无法进行可靠的分组交叉验证。")

    if y.isna().any():
        bad_labels = sorted(set(y_raw[y.isna()].tolist()))
        raise ValueError(
            f"[数据] 标签存在 NaN 或不在 {LABEL_MAP} 中的取值: {bad_labels}"
        )

    X = X.apply(pd.to_numeric, errors="coerce")
    X_array = X.to_numpy(dtype=float)

    if np.isinf(X_array).any():
        bad_cols = X.columns[np.isinf(X_array).any(axis=0)].tolist()
        raise ValueError(f"[数据] 特征存在 inf，问题列示例: {bad_cols[:10]}")

    # sklearn 的 LinearSVC 不接受 NaN；不在此脚本中单独插补，
    # 避免线性支持向量机和 XGBoost 使用不同的数据处理方式。
    if X.isna().any().any():
        n_nan = int(X.isna().sum().sum())
        bad_cols = X.columns[X.isna().any(axis=0)].tolist()
        raise ValueError(
            f"[数据] 特征中存在 {n_nan} 个 NaN，问题列示例: {bad_cols[:10]}。"
            "请先对所有待比较模型使用同一套缺失值处理，再运行本脚本。"
        )

    groups_arr = groups.to_numpy()
    feature_names = X.columns.tolist()

    print(f"原始标签类别: {np.unique(y_raw)} -> 映射 {LABEL_MAP}")
    print(f"数据维度: X={X.shape}, y={y.shape}")
    print(f"正负样本数 (0/1): {(y == 0).sum()} / {(y == 1).sum()}")
    print(f"分组数(protein_id): {len(np.unique(groups_arr))}")

    return {
        "X": X,
        "y": y.astype(int),
        "y_raw": y_raw,
        "groups": groups,
        "groups_arr": groups_arr,
        "feature_names": feature_names,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_ratio": {
            "neg_0": int((y == 0).sum()),
            "pos_1": int((y == 1).sum()),
        },
    }


def assert_no_group_overlap(group_a, group_b, context, log_path=None):
    """确保两个数据子集没有共享 protein_id。"""
    set_a = set(np.unique(np.asarray(group_a)).tolist())
    set_b = set(np.unique(np.asarray(group_b)).tolist())
    overlap = set_a.intersection(set_b)
    line = (
        f"[泄漏检查][{context}] |A|={len(set_a)} |B|={len(set_b)} "
        f"交集={len(overlap)}"
    )
    print("  " + line)

    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if overlap:
        raise ValueError(
            f"[数据泄漏] {context} 存在 {len(overlap)} 个重复 protein_id，"
            f"示例: {sorted(list(overlap))[:10]}"
        )


def make_group_aware_inner_split(
    y_train_full,
    groups_train_full,
    base_seed,
    max_retries=None,
):
    """与 XGBoost 脚本一致的组感知内部 train/validation 划分。"""
    if max_retries is None:
        max_retries = INNER_MAX_RETRIES
    y_array = np.asarray(y_train_full)
    group_array = np.asarray(groups_train_full)
    all_idx = np.arange(len(y_array))
    n_groups = len(np.unique(group_array))

    if n_groups < 2:
        raise ValueError(
            f"[内部划分] 训练集分组数={n_groups} < 2，无法进行 group-aware 划分。"
        )

    for retry in range(max_retries):
        seed = int(base_seed) + retry
        train_idx = val_idx = None

        if HAS_SGKF:
            try:
                n_splits = max(2, min(INNER_N_SPLITS, n_groups))
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=seed,
                )
                train_idx, val_idx = next(
                    splitter.split(all_idx, y_array, group_array)
                )
            except Exception:
                train_idx = val_idx = None

        if train_idx is None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=INNER_VALID_SIZE,
                random_state=seed,
            )
            train_idx, val_idx = next(
                splitter.split(all_idx, y_array, group_array)
            )

        if (
            len(np.unique(y_array[train_idx])) == 2
            and len(np.unique(y_array[val_idx])) == 2
        ):
            return train_idx, val_idx

    raise ValueError(
        f"[内部划分] 重试 {max_retries} 次后，仍无法得到 train/validation "
        f"均包含两类的分组划分。当前训练折分组数={n_groups}。"
    )


def select_outer_fold_features(X_train_full, fold, outdir):
    """仅根据 outer training fold 删除近常数/零方差特征。"""
    if not DROP_NEAR_CONSTANT:
        return list(X_train_full.columns), []

    std = X_train_full.std(axis=0).fillna(0.0)
    n_unique = X_train_full.nunique(dropna=False)
    removed = X_train_full.columns[
        (std < NEAR_CONST_STD_TOL) | (n_unique <= 1)
    ].tolist()
    kept = [column for column in X_train_full.columns if column not in removed]

    if not kept:
        kept = list(X_train_full.columns)
        removed = []

    removed_path = os.path.join(outdir, f"fold{fold}_removed_features.txt")
    with open(removed_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"# fold{fold} 仅根据 outer training fold 删除的近常数特征\n"
        )
        handle.write(
            f"# 规则: std < {NEAR_CONST_STD_TOL} 或唯一值数量 <= 1\n"
        )
        handle.write("\n".join(map(str, removed)) + "\n" if removed else "（无）\n")

    if removed:
        print(
            f"  [特征] fold{fold} 删除近常数列 {len(removed)} 个，"
            f"剩余 {len(kept)} 个"
        )

    return kept, removed


def sample_param_grid(n_trials, seed):
    """生成不重复的 LinearSVC 参数组合；trial 0 固定为基线。"""
    n_trials = max(1, int(n_trials))
    rng = np.random.RandomState(int(seed))

    all_candidates = [
        {"C": float(c_value), "class_weight": class_weight}
        for c_value in SEARCH_SPACE["C"]
        for class_weight in SEARCH_SPACE["class_weight"]
    ]

    baseline = dict(BASELINE_PARAMS)
    remaining = [
        params for params in all_candidates
        if not (
            np.isclose(params["C"], baseline["C"])
            and params["class_weight"] == baseline["class_weight"]
        )
    ]
    rng.shuffle(remaining)
    ordered = [baseline] + remaining

    if n_trials > len(ordered):
        print(
            f"  [参数] 请求 {n_trials} 个 trial，但搜索空间仅有 "
            f"{len(ordered)} 个不重复组合；将运行全部组合。"
        )

    return ordered[:min(n_trials, len(ordered))]


def make_model(params):
    """构建单一 LinearSVC；样本数远大于特征数时使用 primal 优化。"""
    return LinearSVC(
        C=float(params["C"]),
        class_weight=params["class_weight"],
        penalty=FIXED_MODEL_PARAMS["penalty"],
        loss=FIXED_MODEL_PARAMS["loss"],
        dual=FIXED_MODEL_PARAMS["dual"],
        tol=FIXED_MODEL_PARAMS["tol"],
        max_iter=FIXED_MODEL_PARAMS["max_iter"],
        random_state=RANDOM_SEED,
    )


def save_confusion_matrix(cm, fold, outdir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[0, 1],
        yticklabels=[0, 1],
    )
    plt.title(f"Fold {fold} Confusion Matrix (Linear SVM)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"fold{fold}_linear_svm_cm.png"),
        dpi=150,
    )
    plt.close()


def save_roc_curve(y_true, decision_score, fold, outer_auc, outdir):
    fpr, tpr, _ = roc_curve(y_true, decision_score)
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        linewidth=2,
        label=f"ROC curve (AUC = {outer_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Fold {fold} ROC Curve (Linear SVM)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"fold{fold}_linear_svm_roc.png"),
        dpi=150,
    )
    plt.close()


def run_one_fold(fold, data, outer_train_idx, outer_test_idx, outdir, leak_log):
    """完成一个外层折：内部选参，然后外层测试一次。"""
    X = data["X"]
    y = data["y"]
    groups_arr = data["groups_arr"]

    X_train_full = X.iloc[outer_train_idx].copy()
    y_train_full = y.iloc[outer_train_idx].copy()
    groups_train_full = groups_arr[outer_train_idx]

    X_test = X.iloc[outer_test_idx].copy()
    y_test = y.iloc[outer_test_idx].copy()
    groups_test = groups_arr[outer_test_idx]

    assert_no_group_overlap(
        groups_train_full,
        groups_test,
        f"fold{fold} outer train/test",
        leak_log,
    )

    if len(np.unique(y_train_full)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            f"[Fold {fold}] outer train 或 outer test 为单类，无法计算 AUC。"
        )

    kept_columns, removed_columns = select_outer_fold_features(
        X_train_full,
        fold,
        outdir,
    )
    X_train_full = X_train_full[kept_columns]
    X_test = X_test[kept_columns]

    base_seed = RANDOM_SEED + 1000 * fold
    inner_train_idx, inner_val_idx = make_group_aware_inner_split(
        y_train_full,
        groups_train_full,
        base_seed,
    )

    X_train = X_train_full.iloc[inner_train_idx]
    y_train = y_train_full.iloc[inner_train_idx]
    X_val = X_train_full.iloc[inner_val_idx]
    y_val = y_train_full.iloc[inner_val_idx]

    assert_no_group_overlap(
        groups_train_full[inner_train_idx],
        groups_train_full[inner_val_idx],
        f"fold{fold} inner train/val",
        leak_log,
    )

    parameter_grid = sample_param_grid(N_TRIALS, RANDOM_SEED + fold)
    trial_records = []

    # 关键：调参循环只允许查看 inner validation，绝不查看 outer test。
    for trial_index, params in enumerate(parameter_grid):
        model = make_model(params)
        model.fit(X_train, y_train)
        val_score = model.decision_function(X_val)
        val_auc = float(roc_auc_score(y_val, val_score))

        record = {
            "trial": int(trial_index),
            "is_baseline": bool(trial_index == 0),
            "inner_val_auc": val_auc,
            "n_iter": int(model.n_iter_),
            **params,
        }
        trial_records.append(record)

        if trial_index == 0 or (trial_index + 1) % 10 == 0:
            print(
                f"  [fold{fold}] trial {trial_index + 1}/{len(parameter_grid)} "
                f"inner_val_auc={val_auc:.4f}"
            )

    trials_df = pd.DataFrame(trial_records)
    trials_df = trials_df.sort_values(
        ["inner_val_auc", "trial"],
        ascending=[False, True],
    ).reset_index(drop=True)
    trials_df.to_csv(
        os.path.join(outdir, f"fold{fold}_linear_svm_hyperparam_trials.csv"),
        index=False,
    )

    best_row = trials_df.iloc[0].to_dict()
    best_params = {name: best_row[name] for name in SEARCH_SPACE}

    # pandas 会把 class_weight=None 写成 NaN，这里恢复 sklearn 所需类型。
    best_params["C"] = float(best_params["C"])
    if pd.isna(best_params["class_weight"]):
        best_params["class_weight"] = None
    else:
        best_params["class_weight"] = to_builtin(best_params["class_weight"])

    inner_val_auc = float(best_row["inner_val_auc"])

    with open(
        os.path.join(outdir, f"fold{fold}_best_params.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "fold": int(fold),
                "best_params": best_params,
                "inner_val_auc": inner_val_auc,
                "selected_baseline": bool(best_row["is_baseline"]),
                "n_kept_features": int(len(kept_columns)),
                "n_removed_features": int(len(removed_columns)),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    # 与当前 XGBoost 折内协议一致：仍在 inner train 上拟合最佳参数模型。
    best_model = make_model(best_params)
    best_model.fit(X_train, y_train)

    # outer test 至此才第一次被模型用于评价；每折只评价一次。
    test_score = best_model.decision_function(X_test)
    test_prediction = best_model.predict(X_test).astype(int)

    metrics = {
        "outer_test_auc": float(roc_auc_score(y_test, test_score)),
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "f1_score": float(f1_score(y_test, test_prediction, zero_division=0)),
        "precision": float(
            precision_score(y_test, test_prediction, zero_division=0)
        ),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
    }

    print(
        f"  Fold {fold} -> inner_val_AUC={inner_val_auc:.4f} | "
        f"outer_test_AUC={metrics['outer_test_auc']:.4f}, "
        f"ACC={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, "
        f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}"
    )

    model_path = os.path.join(
        outdir,
        f"fold{fold}_linear_svm_best_model.joblib",
    )
    joblib.dump(
        {
            "model": best_model,
            "features": kept_columns,
            "label_map": LABEL_MAP,
            "inverse_label_map": INVERSE_LABEL_MAP,
            "best_params": best_params,
            "inner_val_auc": inner_val_auc,
            "outer_test_auc": metrics["outer_test_auc"],
            "note": "折内模型仅用于交叉验证复现；论文指标取 cv_linear_svm_results.csv。",
        },
        model_path,
    )

    prediction_df = X_test.copy()
    prediction_df.insert(0, "protein_id", np.asarray(groups_test))
    prediction_df["true_label"] = (
        pd.Series(y_test.to_numpy()).map(INVERSE_LABEL_MAP).to_numpy()
    )
    prediction_df["pred_label"] = (
        pd.Series(test_prediction).map(INVERSE_LABEL_MAP).to_numpy()
    )
    prediction_df["decision_score"] = test_score
    prediction_df.to_csv(
        os.path.join(outdir, f"fold{fold}_linear_svm_predictions.csv"),
        index=False,
    )

    cm = confusion_matrix(y_test, test_prediction)
    save_confusion_matrix(cm, fold, outdir)
    save_roc_curve(
        y_test,
        test_score,
        fold,
        metrics["outer_test_auc"],
        outdir,
    )

    return {
        "fold": int(fold),
        "inner_val_auc": inner_val_auc,
        "best_params": best_params,
        "selected_baseline": bool(best_row["is_baseline"]),
        "n_removed_features": int(len(removed_columns)),
        **metrics,
    }


def run_cv(data, outdir):
    """运行与 XGBoost 相同的外层 5-fold GroupKFold。"""
    os.makedirs(outdir, exist_ok=True)

    leak_log = os.path.join(outdir, "group_leakage_checks.txt")
    with open(leak_log, "w", encoding="utf-8") as handle:
        handle.write(
            "# outer train/test 与 inner train/validation 的 protein_id 交集均应为 0\n"
        )

    splitter = GroupKFold(n_splits=N_FOLDS)
    results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data["X"], data["y"], groups=data["groups"]),
        start=1,
    ):
        print(f"\n======= 第 {fold} 折 / 共 {N_FOLDS} 折 =======")
        result = run_one_fold(
            fold,
            data,
            train_idx,
            test_idx,
            outdir,
            leak_log,
        )
        results.append(
            {
                "fold": result["fold"],
                "outer_test_auc": result["outer_test_auc"],
                "inner_val_auc": result["inner_val_auc"],
                "accuracy": result["accuracy"],
                "f1_score": result["f1_score"],
                "precision": result["precision"],
                "recall": result["recall"],
                "n_removed_features": result["n_removed_features"],
                "selected_baseline": result["selected_baseline"],
                "best_params": json.dumps(
                    result["best_params"],
                    ensure_ascii=False,
                ),
            }
        )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(outdir, "cv_linear_svm_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n========== 交叉验证汇总 (mean ± std) ==========")
    for column in [
        "outer_test_auc",
        "inner_val_auc",
        "accuracy",
        "f1_score",
        "precision",
        "recall",
    ]:
        print(
            f"{column:>16}: {results_df[column].mean():.4f} "
            f"± {results_df[column].std():.4f}"
        )

    return results_df


def train_final_model(data, cv_df, outdir):
    """训练全量线性支持向量机，仅供保存/解释，不参与 CV 性能报告。"""
    if cv_df.empty:
        return None

    best_fold_row = cv_df.sort_values(
        ["inner_val_auc", "fold"],
        ascending=[False, True],
    ).iloc[0]
    best_params = json.loads(best_fold_row["best_params"])

    final_model = make_model(best_params)
    final_model.fit(data["X"], data["y"])

    model_path = os.path.join(outdir, "linear_svm_final_model.joblib")
    joblib.dump(
        {
            "model": final_model,
            "features": data["feature_names"],
            "label_map": LABEL_MAP,
            "inverse_label_map": INVERSE_LABEL_MAP,
            "best_params": best_params,
            "from_fold": int(best_fold_row["fold"]),
            "trained_on_all_samples": True,
            "deployment_only": True,
            "note": "全量模型仅供保存/解释；不得用其训练集分数代替外层 CV 指标。",
        },
        model_path,
    )

    with open(
        os.path.join(outdir, "final_best_params.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "best_params": best_params,
                "from_fold": int(best_fold_row["fold"]),
                "source_inner_val_auc": float(best_fold_row["inner_val_auc"]),
                "trained_on_all_samples": True,
                "deployment_only": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"最终全量线性支持向量机已保存: {model_path}")
    return final_model


def save_summary(data, cv_df, outdir, elapsed_seconds):
    """保存论文比较所需的均值、标准差和配置。"""
    def mean_std(column):
        return float(cv_df[column].mean()), float(cv_df[column].std())

    mean_auc, std_auc = mean_std("outer_test_auc")
    mean_inner_auc, std_inner_auc = mean_std("inner_val_auc")
    mean_accuracy, std_accuracy = mean_std("accuracy")
    mean_f1, std_f1 = mean_std("f1_score")
    mean_precision, std_precision = mean_std("precision")
    mean_recall, std_recall = mean_std("recall")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "single LinearSVC (group-aware nested parameter search)",
        "primary_metric": f"mean outer_test_auc across {N_FOLDS} GroupKFold folds",
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_inner_val_auc": mean_inner_auc,
        "std_inner_val_auc": std_inner_auc,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "n_folds_evaluated": int(len(cv_df)),
        "data": {
            "data_path": DATA_REFERENCE,
            "standardized_input": True,
            "n_samples": data["n_samples"],
            "n_features": data["n_features"],
            "n_groups": int(len(np.unique(data["groups_arr"]))),
            "class_ratio_0_1": data["class_ratio"],
        },
        "config": {
            "RANDOM_SEED": RANDOM_SEED,
            "N_FOLDS": N_FOLDS,
            "N_TRIALS": N_TRIALS,
            "INNER_N_SPLITS": INNER_N_SPLITS,
            "INNER_VALID_SIZE": INNER_VALID_SIZE,
            "INNER_MAX_RETRIES": INNER_MAX_RETRIES,
            "DROP_NEAR_CONSTANT": DROP_NEAR_CONSTANT,
            "NEAR_CONST_STD_TOL": NEAR_CONST_STD_TOL,
            "inner_split_method": (
                "StratifiedGroupKFold" if HAS_SGKF else "GroupShuffleSplit"
            ),
            "search_space": SEARCH_SPACE,
            "baseline_params": BASELINE_PARAMS,
        },
        "per_fold": cv_df.to_dict(orient="records"),
        "elapsed_sec": round(float(elapsed_seconds), 2),
    }

    with open(
        os.path.join(outdir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def run_training():
    start_time = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 72)
    print("单一 LinearSVC + 组感知内部调参")
    print(f"输入数据: {DATA_REFERENCE}")
    print(f"外层折数: {N_FOLDS} | 每折参数 trial: {N_TRIALS}")
    print(
        "内部划分: "
        + ("StratifiedGroupKFold" if HAS_SGKF else "GroupShuffleSplit")
    )
    print("选参指标: inner validation AUC")
    print("论文比较指标: mean outer GroupKFold test AUC ± std")
    print(f"输出目录: {OUTDIR}")
    print("=" * 72)

    data = load_data(DATA_PATH)
    cv_df = run_cv(data, OUTDIR)
    train_final_model(data, cv_df, OUTDIR)
    summary = save_summary(
        data,
        cv_df,
        OUTDIR,
        time.time() - start_time,
    )

    print("\n" + "=" * 72)
    print("Comparable Linear SVM training finished.")
    print(
        f"Mean outer GroupKFold AUC = {summary['mean_auc']:.4f} "
        f"± {summary['std_auc']:.4f}"
    )
    print(f"Results saved to: {OUTDIR}")
    print("=" * 72)
    return summary


def main(argv=None):
    """Run with CLI overrides; importing this module does not start training."""
    return run_cli(
        script_file=__file__,
        model_name="linear_svm",
        namespace=globals(),
        training_function=run_training,
        argv=argv,
    )


if __name__ == "__main__":
    main()
