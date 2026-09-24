#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2 linear_svm: protein-cluster bootstrap of saved outer-test predictions.

Input labels are the ORIGINAL 1/2 labels. Internally, 1 -> 0 and 2 -> 1;
original label 2 is the positive class. Resampling, classification rules,
fold averaging and percentile calculations are retained from the source.
This analysis does not refit models and does not compute SHAP values.

Run from the repository root:
    python scripts/evaluation/task2/bootstrap/bootstrap_ci_linear_svm.py --result-dir PATH
Default settings: configs/task2/evaluation/bootstrap/linear_svm.json.
Outputs default to <result-dir>/bootstrap; populated directories are refused.
"""

from __future__ import annotations

import sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from _eval_runtime import bootstrap_args, public_path, save_run_info

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EXPECTED_FOLDS = (1, 2, 3, 4, 5)
SOURCE_SCORE_COLUMN = "decision_score"

REQUIRED_COLUMNS = {
    "protein_id",
    "true_label",
    "pred_label",
    SOURCE_SCORE_COLUMN,
}
METRIC_NAMES = (
    "roc_auc",
    "accuracy",
    "f1_binary_label2",
    "f1_macro",
    "precision_label2",
    "recall_label2",
)
CV_REQUIRED_COLUMNS = {
    "fold",
    "outer_test_auc",
    "accuracy",
    "f1_score",
    "precision",
    "recall",
}
CONSISTENCY_TOL = 1e-8


def parse_args(argv=None) -> argparse.Namespace:
    args = bootstrap_args(__file__, "linear_svm", argv)
    global CONSISTENCY_TOL
    CONSISTENCY_TOL = args.consistency_tol
    return args


def discover_prediction_files(result_dir: Path) -> List[Tuple[int, Path]]:
    """发现 5 个 Linear SVM outer-test 预测文件，并核对折编号。"""
    candidates = list(result_dir.glob("fold*linear_svm_predictions.csv"))
    pattern = re.compile(
        r"fold_?(\d+)_linear_svm_predictions\.csv$",
        re.IGNORECASE,
    )
    found: List[Tuple[int, Path]] = []

    for path in candidates:
        match = pattern.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))

    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(
            f"在 {result_dir} 中未找到 fold1_linear_svm_predictions.csv 等文件。"
        )

    fold_ids = [fold for fold, _ in found]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"检测到重复 fold 编号：{fold_ids}")

    if tuple(fold_ids) != EXPECTED_FOLDS:
        raise ValueError(
            f"应当找到 5 个 outer folds：{list(EXPECTED_FOLDS)}；"
            f"实际找到：{fold_ids}。"
        )
    return found


def read_prediction_columns(path: Path) -> pd.DataFrame:
    """
    只读取 Bootstrap 所需的 4 列，避免载入预测 CSV 中的数百个特征列。

    Linear SVM 训练脚本直接保存 decision_score；
    ROC-AUC 使用该连续分数计算，分类指标使用保存的 pred_label 计算。
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(header)
    if missing:
        raise ValueError(f"{path.name} 缺少必要列：{sorted(missing)}")

    return pd.read_csv(
        path,
        usecols=[
            "protein_id",
            "true_label",
            "pred_label",
            SOURCE_SCORE_COLUMN,
        ],
    )


def map_original_1_2_to_binary(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """明确执行原始标签 1->0、2->1。"""
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column_name} 中存在缺失值或非数值。")

    values = values.astype(int)
    unique_values = set(values.unique().tolist())
    if not unique_values.issubset({1, 2}):
        raise ValueError(
            f"{column_name} 必须为原始标签 1/2；"
            f"实际发现：{sorted(unique_values)}。"
        )
    return values.map({1: 0, 2: 1}).astype(np.int8)


def load_and_validate_predictions(
    result_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取并严格验证预测文件。

    fin_1_2_svm.py 明确使用：
        test_score = best_model.decision_function(X_test)
        test_prediction = best_model.predict(X_test).astype(int)

    对二分类 LinearSVC，sklearn 的 predict 规则等价于：
        decision_score > 0  -> 模型类别1（原始标签2）
        decision_score <= 0 -> 模型类别0（原始标签1）

    后续 Accuracy/F1/Precision/Recall 始终使用保存的 pred_label；
    decision_score 仅用于 ROC-AUC 及预测文件内部一致性检查。
    一旦发现不一致，本脚本直接停止。
    """
    prediction_files = discover_prediction_files(result_dir)
    frames: List[pd.DataFrame] = []

    for fold, path in prediction_files:
        print(f"读取 {path.name}（仅载入 4 个必要列）……")
        part = read_prediction_columns(path)
        part.insert(0, "fold", fold)
        part["source_file"] = path.name
        frames.append(part)

    all_pred = pd.concat(frames, ignore_index=True)

    if all_pred["protein_id"].isna().any():
        raise ValueError(
            "protein_id 中存在缺失值，无法执行 protein-cluster Bootstrap。"
        )

    all_pred["y_true"] = map_original_1_2_to_binary(
        all_pred["true_label"],
        "true_label",
    )
    all_pred["y_pred_saved"] = map_original_1_2_to_binary(
        all_pred["pred_label"],
        "pred_label",
    )

    all_pred["decision_score"] = pd.to_numeric(
        all_pred["decision_score"],
        errors="coerce",
    )
    if all_pred["decision_score"].isna().any():
        raise ValueError("decision_score 中存在缺失值或非数值。")
    if not np.isfinite(
        all_pred["decision_score"].to_numpy(dtype=float)
    ).all():
        raise ValueError("decision_score 中存在 inf 或 -inf。")

    # LinearSVC.predict 对二分类使用严格的正决策分数；
    # 因而概率恰好为 0.5 时归为模型类别0（原始标签1）。
    all_pred["y_pred_from_probability"] = (
        all_pred["decision_score"] > 0.0
    ).astype(np.int8)

    mismatch_mask = (
        all_pred["y_pred_saved"].to_numpy()
        != all_pred["y_pred_from_probability"].to_numpy()
    )
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count > 0:
        mismatch_rate = mismatch_count / len(all_pred)
        examples = all_pred.loc[
            mismatch_mask,
            [
                "fold",
                "protein_id",
                "true_label",
                "pred_label",
                "decision_score",
            ],
        ].head(10).copy()
        examples["expected_pred_label_from_linear_svm_score_rule"] = np.where(
            examples["decision_score"].to_numpy() > 0.0,
            2,
            1,
        )
        raise ValueError(
            "\n预测文件内部不一致，Bootstrap 已中止。\n"
            f"pred_label 与 LinearSVC 的 decision_score>0 规则不一致："
            f"{mismatch_count}/{len(all_pred)} 行 "
            f"({mismatch_rate:.2%})。\n"
            "按照上传的 Linear SVM 训练脚本，这两个结果必须完全一致；"
            "请检查是否混入不同批次文件、预测列是否被修改，"
            "或实际运行的训练脚本是否不是当前版本。\n"
            "前 10 个不一致示例：\n"
            + examples.to_string(index=False)
        )

    # 两者已严格一致，后续统一使用 y_pred。
    all_pred["y_pred"] = all_pred["y_pred_saved"].astype(np.int8)
    print("检查通过：pred_label 与 LinearSVC 的 decision_score 决策规则完全一致。")

    # 同一 protein_id 只能属于一个 outer-test fold。
    folds_per_protein = all_pred.groupby(
        "protein_id",
        dropna=False,
    )["fold"].nunique()
    leaked = folds_per_protein[folds_per_protein > 1]
    if not leaked.empty:
        raise ValueError(
            "同一 protein_id 出现在多个 outer-test folds，"
            "说明分组交叉验证输出存在泄漏或文件混用。"
            f"示例：{leaked.index[:10].tolist()}"
        )

    # 每个 outer fold 均必须包含两个真实类别。
    for fold, block in all_pred.groupby("fold", sort=True):
        if block["y_true"].nunique() != 2:
            raise ValueError(
                f"fold {fold} 的 outer-test 预测只包含一个真实类别，"
                "无法计算 ROC-AUC。"
            )

    return all_pred, [path.name for _, path in prediction_files]


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_confusion_and_auc(
    tn: float,
    fp: float,
    fn: float,
    tp: float,
    auc: float,
) -> Dict[str, float]:
    total = tn + fp + fn + tp
    f1_positive = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    f1_negative = safe_divide(2.0 * tn, 2.0 * tn + fp + fn)

    return {
        "roc_auc": float(auc),
        "accuracy": safe_divide(tp + tn, total),
        "f1_binary_label2": f1_positive,
        "f1_macro": 0.5 * (f1_positive + f1_negative),
        "precision_label2": safe_divide(tp, tp + fp),
        "recall_label2": safe_divide(tp, tp + fn),
    }


@dataclass
class FoldStats:
    """一个 outer fold 的蛋白簇级充分统计量。"""

    fold: int
    group_ids: np.ndarray
    n_rows: np.ndarray
    positive_count: np.ndarray
    negative_count: np.ndarray
    tp_count: np.ndarray
    fp_count: np.ndarray
    tn_count: np.ndarray
    fn_count: np.ndarray
    auc_pair_contribution: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(len(self.group_ids))

    def evaluate_multiplicity(
        self,
        multiplicity: np.ndarray,
    ) -> Dict[str, float]:
        """
        根据每个 protein_id 在一次 Bootstrap 中被抽中的次数计算指标。
        与实际复制该蛋白全部帧后再计算指标等价。
        """
        m = np.asarray(multiplicity, dtype=float)
        if m.shape != (self.n_groups,):
            raise ValueError(
                f"fold {self.fold}: multiplicity 形状应为 "
                f"{(self.n_groups,)}，实际为 {m.shape}。"
            )

        n_positive = float(m @ self.positive_count)
        n_negative = float(m @ self.negative_count)
        if n_positive <= 0 or n_negative <= 0:
            raise ValueError(
                f"fold {self.fold}: Bootstrap 样本缺少一个真实类别。"
            )

        # AUC 分子：
        # protein i 中全部正例与 protein j 中全部负例之间的胜出数
        # + 0.5 × 平局数，再乘各蛋白簇的 Bootstrap 重数。
        auc_numerator = float(
            np.einsum(
                "i,ij,j->",
                m,
                self.auc_pair_contribution,
                m,
            )
        )
        auc = auc_numerator / (n_positive * n_negative)

        tp = float(m @ self.tp_count)
        fp = float(m @ self.fp_count)
        tn = float(m @ self.tn_count)
        fn = float(m @ self.fn_count)

        return metrics_from_confusion_and_auc(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            auc=auc,
        )


def build_fold_stats(df_fold: pd.DataFrame) -> FoldStats:
    """预计算一个 outer fold 的蛋白簇混淆计数与 AUC 蛋白对贡献。"""
    fold = int(df_fold["fold"].iloc[0])
    group_ids = pd.unique(df_fold["protein_id"])
    n_groups = len(group_ids)

    n_rows = np.zeros(n_groups, dtype=np.int64)
    positive_count = np.zeros(n_groups, dtype=np.int64)
    negative_count = np.zeros(n_groups, dtype=np.int64)
    tp_count = np.zeros(n_groups, dtype=np.int64)
    fp_count = np.zeros(n_groups, dtype=np.int64)
    tn_count = np.zeros(n_groups, dtype=np.int64)
    fn_count = np.zeros(n_groups, dtype=np.int64)

    positive_scores: List[np.ndarray] = []
    negative_scores_sorted: List[np.ndarray] = []

    grouped = df_fold.groupby("protein_id", sort=False, dropna=False)
    for group_index, group_id in enumerate(group_ids):
        block = grouped.get_group(group_id)
        y_true = block["y_true"].to_numpy(dtype=np.int8)
        y_pred = block["y_pred"].to_numpy(dtype=np.int8)
        scores = block["decision_score"].to_numpy(dtype=float)

        n_rows[group_index] = len(block)
        positive_count[group_index] = int((y_true == 1).sum())
        negative_count[group_index] = int((y_true == 0).sum())
        tp_count[group_index] = int(
            ((y_true == 1) & (y_pred == 1)).sum()
        )
        fp_count[group_index] = int(
            ((y_true == 0) & (y_pred == 1)).sum()
        )
        tn_count[group_index] = int(
            ((y_true == 0) & (y_pred == 0)).sum()
        )
        fn_count[group_index] = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        positive_scores.append(scores[y_true == 1])
        negative_scores_sorted.append(np.sort(scores[y_true == 0]))

    auc_pair_contribution = np.zeros(
        (n_groups, n_groups),
        dtype=np.float64,
    )

    # A[i, j] = protein i 的正例分数与 protein j 的负例分数比较时：
    # 正例分数更高计 1，平局计 0.5。
    for i, positive in enumerate(positive_scores):
        if positive.size == 0:
            continue
        for j, negative_sorted in enumerate(negative_scores_sorted):
            if negative_sorted.size == 0:
                continue

            n_lower = np.searchsorted(
                negative_sorted,
                positive,
                side="left",
            )
            n_lower_or_equal = np.searchsorted(
                negative_sorted,
                positive,
                side="right",
            )
            auc_pair_contribution[i, j] = float(
                np.sum(
                    n_lower
                    + 0.5 * (n_lower_or_equal - n_lower),
                    dtype=np.float64,
                )
            )

    stats = FoldStats(
        fold=fold,
        group_ids=np.asarray(group_ids),
        n_rows=n_rows,
        positive_count=positive_count,
        negative_count=negative_count,
        tp_count=tp_count,
        fp_count=fp_count,
        tn_count=tn_count,
        fn_count=fn_count,
        auc_pair_contribution=auc_pair_contribution,
    )

    # 自检：全部蛋白重数为 1 时，预计算算法必须与 sklearn 完全一致。
    multiplicity = np.ones(stats.n_groups, dtype=int)
    precomputed = stats.evaluate_multiplicity(multiplicity)

    y_true = df_fold["y_true"].to_numpy(dtype=int)
    y_pred = df_fold["y_pred"].to_numpy(dtype=int)
    y_probability = df_fold["decision_score"].to_numpy(dtype=float)
    sklearn_reference = {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_binary_label2": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_label2": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_label2": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
    }

    max_difference = max(
        abs(precomputed[name] - sklearn_reference[name])
        for name in METRIC_NAMES
    )
    if max_difference > 1e-10:
        raise RuntimeError(
            f"fold {fold}: Bootstrap 预计算算法与 sklearn 指标不一致，"
            f"最大差异={max_difference:.12g}。"
        )

    return stats


def build_all_fold_stats(
    all_predictions: pd.DataFrame,
) -> Dict[int, FoldStats]:
    fold_stats: Dict[int, FoldStats] = {}

    for fold, block in all_predictions.groupby("fold", sort=True):
        print(
            f"预计算 fold {int(fold)}："
            f"{block['protein_id'].nunique()} 个 protein_id，"
            f"{len(block)} 帧……"
        )
        fold_stats[int(fold)] = build_fold_stats(block)

    if tuple(sorted(fold_stats)) != EXPECTED_FOLDS:
        raise ValueError(
            f"预计算后的 fold 编号异常：{sorted(fold_stats)}。"
        )
    return fold_stats


def recompute_fold_metrics(
    fold_stats: Dict[int, FoldStats],
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    for fold in sorted(fold_stats):
        stats = fold_stats[fold]
        record: Dict[str, float] = {
            "fold": int(fold),
            "n_rows": int(stats.n_rows.sum()),
            "n_proteins": int(stats.n_groups),
        }
        record.update(
            stats.evaluate_multiplicity(
                np.ones(stats.n_groups, dtype=int)
            )
        )
        records.append(record)

    return pd.DataFrame(records)


def validate_against_cv_results(
    result_dir: Path,
    fold_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    将预测文件重算指标与 cv_linear_svm_results.csv 逐折严格核对。
    若任一指标差异超过 1e-8，立即中止，不生成 Bootstrap 结论。
    """
    cv_path = result_dir / "cv_linear_svm_results.csv"
    if not cv_path.exists():
        raise FileNotFoundError(
            f"未找到 {cv_path.name}。"
            "为了防止预测文件与训练汇总混用，本脚本要求该文件必须存在。"
        )

    cv_results = pd.read_csv(cv_path)
    missing = CV_REQUIRED_COLUMNS.difference(cv_results.columns)
    if missing:
        raise ValueError(
            f"{cv_path.name} 缺少必要列：{sorted(missing)}。"
        )

    cv_results = cv_results[
        [
            "fold",
            "outer_test_auc",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
        ]
    ].copy()
    cv_results["fold"] = pd.to_numeric(
        cv_results["fold"],
        errors="raise",
    ).astype(int)

    if tuple(sorted(cv_results["fold"].tolist())) != EXPECTED_FOLDS:
        raise ValueError(
            f"{cv_path.name} 中的 fold 编号应为 {list(EXPECTED_FOLDS)}，"
            f"实际为 {sorted(cv_results['fold'].tolist())}。"
        )

    # 先给 CV 汇总列加前缀，避免与重算指标在 merge 后发生 _x/_y 冲突。
    cv_results = cv_results.rename(
        columns={
            "outer_test_auc": "cv_outer_test_auc",
            "accuracy": "cv_accuracy",
            "f1_score": "cv_f1_binary_label2",
            "precision": "cv_precision_label2",
            "recall": "cv_recall_label2",
        }
    )

    comparison = fold_metrics.merge(
        cv_results,
        on="fold",
        how="inner",
        validate="one_to_one",
    )

    pairs = {
        "roc_auc": "cv_outer_test_auc",
        "accuracy": "cv_accuracy",
        "f1_binary_label2": "cv_f1_binary_label2",
        "precision_label2": "cv_precision_label2",
        "recall_label2": "cv_recall_label2",
    }
    max_differences: Dict[str, float] = {}

    for recomputed_column, cv_column in pairs.items():
        difference_column = f"abs_diff_{recomputed_column}"
        comparison[difference_column] = np.abs(
            comparison[recomputed_column].to_numpy(dtype=float)
            - comparison[cv_column].to_numpy(dtype=float)
        )
        max_differences[recomputed_column] = float(
            comparison[difference_column].max()
        )

    failed = {
        metric: difference
        for metric, difference in max_differences.items()
        if difference > CONSISTENCY_TOL
    }
    if failed:
        details = "\n".join(
            f"  {metric}: max abs diff = {difference:.12g}"
            for metric, difference in failed.items()
        )
        raise ValueError(
            "\n预测文件与 cv_linear_svm_results.csv 不一致，"
            "Bootstrap 已中止。\n"
            + details
            + "\n请确认 5 个预测文件和 CV 汇总文件来自同一次训练，"
            "且当前目录中没有混入旧结果。"
        )

    print(
        "检查通过：预测文件重算的 AUC、Accuracy、binary F1、"
        "Precision 和 Recall 与 cv_linear_svm_results.csv 逐折一致。"
    )
    return comparison, max_differences


def draw_valid_multiplicity(
    stats: FoldStats,
    rng: np.random.Generator,
    max_redraw: int,
) -> Tuple[np.ndarray, int]:
    """在一个 outer fold 内按 protein_id 整簇有放回抽样。"""
    probabilities = np.full(
        stats.n_groups,
        1.0 / stats.n_groups,
        dtype=float,
    )

    for redraw_count in range(max_redraw + 1):
        multiplicity = rng.multinomial(
            stats.n_groups,
            probabilities,
        )
        n_positive = float(
            multiplicity @ stats.positive_count
        )
        n_negative = float(
            multiplicity @ stats.negative_count
        )
        if n_positive > 0 and n_negative > 0:
            return multiplicity, redraw_count

    raise RuntimeError(
        f"fold {stats.fold}: 连续重抽 {max_redraw} 次后仍无法同时包含两类。"
        "请检查该 fold 中标签是否集中在极少数 protein_id。"
    )


def bootstrap_fold_mean(
    fold_stats: Dict[int, FoldStats],
    n_bootstrap: int,
    seed: int,
    max_redraw: int,
) -> Tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []
    total_redraws = 0
    fold_ids = sorted(fold_stats)

    for bootstrap_index in range(n_bootstrap):
        fold_results: List[Dict[str, float]] = []

        for fold in fold_ids:
            stats = fold_stats[fold]
            multiplicity, redraws = draw_valid_multiplicity(
                stats=stats,
                rng=rng,
                max_redraw=max_redraw,
            )
            total_redraws += redraws
            fold_results.append(
                stats.evaluate_multiplicity(multiplicity)
            )

        record: Dict[str, float] = {
            "bootstrap_id": bootstrap_index + 1
        }
        for metric in METRIC_NAMES:
            record[metric] = float(
                np.mean(
                    [fold_result[metric] for fold_result in fold_results]
                )
            )
        records.append(record)

        progress_step = max(1, n_bootstrap // 10)
        if (bootstrap_index + 1) % progress_step == 0:
            print(
                f"Bootstrap: {bootstrap_index + 1}/{n_bootstrap}"
            )

    return pd.DataFrame(records), total_redraws


def summarize_confidence_intervals(
    fold_metrics: pd.DataFrame,
    bootstrap_distribution: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    alpha = 1.0 - confidence
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    metric_descriptions = {
        "roc_auc": "ROC-AUC based on LinearSVC decision-score ranking",
        "accuracy": "Frame-level accuracy from saved LinearSVC.predict labels",
        "f1_binary_label2": (
            "Binary F1 with original label 2 as positive class"
        ),
        "f1_macro": "Macro-F1 averaged over original labels 1 and 2",
        "precision_label2": (
            "Precision for original label 2"
        ),
        "recall_label2": "Recall for original label 2",
    }

    records: List[Dict[str, float]] = []
    for metric in METRIC_NAMES:
        point_estimate = float(fold_metrics[metric].mean())
        values = bootstrap_distribution[metric].to_numpy(dtype=float)
        records.append(
            {
                "metric": metric,
                "metric_definition": metric_descriptions[metric],
                "point_estimate_mean_outer_folds": point_estimate,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(
                    np.std(values, ddof=1)
                ),
                "ci_level": float(confidence),
                "ci_lower_percentile": float(
                    np.percentile(values, lower_percentile)
                ),
                "ci_upper_percentile": float(
                    np.percentile(values, upper_percentile)
                ),
                "n_bootstrap": int(len(values)),
            }
        )

    return pd.DataFrame(records)


def build_protein_label_composition(
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """汇总每个 protein_id 内的帧级真实标签组成。"""
    rows: List[Dict[str, float]] = []

    grouped = all_predictions.groupby(
        ["fold", "protein_id"],
        sort=True,
        dropna=False,
    )
    for (fold, protein_id), block in grouped:
        n_label1 = int((block["y_true"] == 0).sum())
        n_label2 = int((block["y_true"] == 1).sum())
        total = n_label1 + n_label2

        rows.append(
            {
                "fold": int(fold),
                "protein_id": protein_id,
                "n_frames": int(total),
                "n_true_label_1": n_label1,
                "n_true_label_2": n_label2,
                "fraction_true_label_2": (
                    float(n_label2 / total)
                    if total > 0
                    else np.nan
                ),
                "contains_both_labels": bool(
                    n_label1 > 0 and n_label2 > 0
                ),
            }
        )

    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(args.effective_config, ensure_ascii=False, indent=2))
        return
    result_dir = args.result_dir.expanduser().resolve()

    if not result_dir.exists():
        raise FileNotFoundError(f"结果目录不存在：{result_dir}")
    if args.n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap 过少；建议至少 500，正式结果建议 2000。"
        )
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("confidence 必须位于 (0, 1)。")
    if args.max_redraw < 0:
        raise ValueError("max_redraw 不能为负数。")

    output_dir = args.output_dir

    print("=" * 76)
    print("Linear SVM outer-test protein-cluster Bootstrap")
    print(f"输入结果目录: {result_dir}")
    print(f"Bootstrap 输出目录: {output_dir}")
    print(f"Bootstrap 次数: {args.n_bootstrap}")
    print(f"随机种子: {args.seed}")
    print(f"置信水平: {args.confidence:.3f}")
    print("=" * 76)

    all_predictions, prediction_files = load_and_validate_predictions(
        result_dir
    )

    protein_composition = build_protein_label_composition(
        all_predictions
    )
    frame_counts = protein_composition["n_frames"]
    n_mixed_proteins = int(
        protein_composition["contains_both_labels"].sum()
    )

    print(
        "合并后的 outer-test 预测："
        f"{len(all_predictions)} 帧，"
        f"{all_predictions['protein_id'].nunique()} 个 protein_id，"
        f"{all_predictions['fold'].nunique()} 个 folds。"
    )
    print(
        "每个 protein_id 的帧数："
        f"min={int(frame_counts.min())}, "
        f"median={float(frame_counts.median()):.1f}, "
        f"max={int(frame_counts.max())}。"
    )
    print(
        "同一 protein_id 内同时包含标签 1 和 2："
        f"{n_mixed_proteins}/{len(protein_composition)} 个。"
    )
    print(
        "Bootstrap 单位：protein_id；同一蛋白的全部帧作为一个簇整体抽样。"
    )

    fold_stats = build_all_fold_stats(all_predictions)
    fold_metrics = recompute_fold_metrics(fold_stats)
    fold_comparison, max_differences = validate_against_cv_results(
        result_dir,
        fold_metrics,
    )

    print("\n原始 5 折均值：")
    for metric in METRIC_NAMES:
        print(
            f"  {metric:>22s} = "
            f"{fold_metrics[metric].mean():.6f}"
        )

    bootstrap_distribution, total_redraws = bootstrap_fold_mean(
        fold_stats=fold_stats,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        max_redraw=args.max_redraw,
    )
    ci_summary = summarize_confidence_intervals(
        fold_metrics=fold_metrics,
        bootstrap_distribution=bootstrap_distribution,
        confidence=args.confidence,
    )

    # 所有检查和计算均成功后再创建输出目录，避免留下误导性的半成品。
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "bootstrap_ci_summary.csv"
    distribution_path = output_dir / "bootstrap_distribution.csv"
    fold_metrics_path = output_dir / "recomputed_fold_metrics.csv"
    composition_path = output_dir / "protein_label_composition.csv"
    metadata_path = output_dir / "bootstrap_metadata.json"

    ci_summary.to_csv(summary_path, index=False)
    bootstrap_distribution.to_csv(
        distribution_path,
        index=False,
    )
    fold_comparison.to_csv(
        fold_metrics_path,
        index=False,
    )
    protein_composition.to_csv(
        composition_path,
        index=False,
    )

    metadata = {
        "model": "LinearSVC",
        "source_training_script": "scripts/training/task2/train_linear_svm.py",
        "result_dir": public_path(result_dir),
        "output_dir": public_path(output_dir),
        "prediction_files": prediction_files,
        "cv_results_file": "cv_linear_svm_results.csv",
        "n_rows_frames": int(len(all_predictions)),
        "n_proteins": int(
            all_predictions["protein_id"].nunique()
        ),
        "n_folds": int(all_predictions["fold"].nunique()),
        "n_proteins_with_both_labels": n_mixed_proteins,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "confidence": float(args.confidence),
        "ci_method": "percentile protein-cluster bootstrap",
        "bootstrap_unit": (
            "protein_id cluster, resampled within each outer fold"
        ),
        "within_protein_labels": (
            "frame-level labels may vary within one protein_id"
        ),
        "point_estimate": (
            "arithmetic mean of the five outer-test fold metrics"
        ),
        "positive_class": (
            "original label 2, mapped to model class 1"
        ),
        "prediction_rule": (
            "LinearSVC.predict; original label 2 is predicted when "
            "decision_score > 0, while decision_score <= 0 maps to "
            "original label 1"
        ),
        "source_score_column": SOURCE_SCORE_COLUMN,
        "pred_label_score_consistency": "passed; zero mismatches",
        "cv_prediction_consistency_tolerance": CONSISTENCY_TOL,
        "maximum_absolute_differences_vs_cv_results": max_differences,
        "total_redraws_due_to_single_class_bootstrap_samples": int(
            total_redraws
        ),
        "implementation": (
            "Exact protein-cluster multiplicity calculation using "
            "precomputed confusion counts and protein-pair AUC ranking "
            "contributions. This is equivalent to duplicating all frames "
            "of each resampled protein_id but avoids repeatedly copying "
            "the large prediction tables."
        ),
        "interpretation": (
            "Post-hoc uncertainty of fixed outer-test predictions under "
            "resampling of protein systems. It accounts for dependence "
            "among frames from the same protein at the resampling level, "
            "but it does not include full model retraining, inner "
            "hyperparameter-search uncertainty, or new-domain external "
            "validation."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBootstrap 置信区间：")
    for row in ci_summary.itertuples(index=False):
        print(
            f"  {row.metric:>22s}: "
            f"{row.point_estimate_mean_outer_folds:.6f} "
            f"[{row.ci_lower_percentile:.6f}, "
            f"{row.ci_upper_percentile:.6f}]"
        )

    print(f"\n全部输出已保存到：{output_dir}")
    print(f"  {summary_path.name}")
    print(f"  {distribution_path.name}")
    print(f"  {fold_metrics_path.name}")
    print(f"  {composition_path.name}")
    print(f"  {metadata_path.name}")


    inputs = [path for _, path in discover_prediction_files(result_dir)]
    cv_path = result_dir / "cv_linear_svm_results.csv"
    if cv_path.is_file():
        inputs.append(cv_path)
    save_run_info(output_dir, args, __file__, inputs)


if __name__ == "__main__":
    main()
