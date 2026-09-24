#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1 XGBoost ablation: remove_contacts.

The input must already have the selected feature group removed. Use the paired
prepare_remove_contacts.py first, or provide --data with your existing ablation CSV.
The supplied group-aware tuning, early stopping, score calculation and final
full-data fit are retained. This script does NOT compute SHAP values.
"""
import os
import sys

# -----------------------------------------------------------------------------
# 防御：同目录下若存在 xgboost.py（旧的集成脚本），它会把真正的 xgboost 包遮蔽，
# 导致 `from xgboost import XGBClassifier` 失败。这里把脚本所在目录从 sys.path 的
# 最前面挪到最后，确保优先导入已安装的 xgboost 包，同时仍能 import 本目录其它文件。
# -----------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_here, 'xgboost.py')):
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here] + [_here]

import json
import time
import random
import warnings

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

# StratifiedGroupKFold 在较老 sklearn 里可能没有，缺失则自动回退到 GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

import matplotlib
matplotlib.use("Agg")              # 无显示环境也能保存图片
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
# 配置区（固定随机种子，保证可复现；N_TRIALS 可配置）
# =============================================================================
import sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from _eval_runtime import default_config, apply_ablation_config, public_path, run_ablation_cli

DEFAULT_CONFIG, CONFIG_PATH = default_config("ablation", "remove_contacts")
apply_ablation_config(globals(), DEFAULT_CONFIG)
# Runtime paths are resolved by run_ablation_cli before any data are loaded.
DATA_PATH = DEFAULT_CONFIG["data"]["ablated_table"]
DATA_REFERENCE = DATA_PATH
OUTDIR = DEFAULT_CONFIG["output"]["root"]
LABEL_MAP = {0: 0, 1: 1}
INVERSE_LABEL_MAP = {0: 0, 1: 1}


# =============================================================================
# 基础辅助函数
# =============================================================================
def _py(v):
    """numpy 标量 -> python 标量，避免 JSON / dict 去重时的类型问题。"""
    return v.item() if isinstance(v, np.generic) else v


def spw_of(y_series):
    """scale_pos_weight 基准 = 负样本数 / 正样本数（仅用给定（训练）标签统计）。"""
    pos = max(int((np.asarray(y_series) == 1).sum()), 1)
    return float((np.asarray(y_series) == 0).sum()) / pos


def best_iteration_of(model):
    """取早停得到的最佳树数；没有则返回 None。"""
    bi = getattr(model, 'best_iteration', None)
    if bi is None:
        return None
    try:
        if isinstance(bi, float) and np.isnan(bi):
            return None
    except TypeError:
        pass
    return int(bi)


def assert_no_group_overlap(g_a, g_b, context, log_path=None):
    """防泄露核心：两组 protein_id 不允许有交集；有则直接 raise ValueError。"""
    set_a = set(np.unique(np.asarray(g_a)).tolist())
    set_b = set(np.unique(np.asarray(g_b)).tolist())
    inter = set_a.intersection(set_b)
    line = f"[泄露检查][{context}] |A|={len(set_a)} |B|={len(set_b)} 交集={len(inter)}"
    print("  " + line)
    if log_path is not None:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    if inter:
        raise ValueError(
            f"[数据泄露] {context} 存在分组(protein_id)重叠，共 {len(inter)} 个，"
            f"示例: {sorted(list(inter))[:10]}。GroupKFold/内部划分被破坏，已中止。"
        )
    return len(inter)


# =============================================================================
# 1) 数据加载（与基线完全一致）+ 健全性 / 防泄露检查
# =============================================================================
def load_data(data_path):
    """读取 CSV：第 1 列=protein_id/group，中间列=features，最后 1 列=原始标签(0/1)。
    标签采用恒等映射 {0:0, 1:1}，保持原始 0/1 定义。同时做 NaN / inf 健全性检查。"""
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1].copy()
    y_raw = df.iloc[:, -1].astype(int).copy()
    groups = df.iloc[:, 0]
    y = y_raw.map(LABEL_MAP)

    # ---- 健全性检查：标签不能有 NaN（含未在 LABEL_MAP 中的值），特征不能有 inf ----
    if y.isna().any():
        bad = sorted(set(y_raw[y.isna()].tolist()))
        raise ValueError(f"[数据] 标签存在 NaN 或不在 {LABEL_MAP} 中的取值: {bad}")
    X_num = X.apply(pd.to_numeric, errors='coerce')
    if np.isinf(X_num.to_numpy(dtype=float)).any():
        cols = X_num.columns[np.isinf(X_num.to_numpy(dtype=float)).any(axis=0)].tolist()
        raise ValueError(f"[数据] 特征存在 inf，问题列示例: {cols[:10]}")
    if X_num.isna().any().any():
        # 仅提示：XGBoost 原生支持 NaN，这里不强行报错，但打印出来便于排查
        n_nan = int(X_num.isna().sum().sum())
        print(f"[数据] 提示：特征中存在 {n_nan} 个 NaN（XGBoost 可原生处理，未做填充）。")
    X = X_num

    feature_names = X.columns.tolist()
    n_samples, n_features = X.shape
    groups_arr = groups.to_numpy()

    print(f"原始标签类别: {np.unique(y_raw)}  ->  映射 {LABEL_MAP}")
    print(f"数据维度: X={X.shape}, y={y.shape}")
    print(f"正负样本比例 (0/1): {(y == 0).sum()} / {(y == 1).sum()}")
    print(f"分组数(protein_id): {len(np.unique(groups_arr))}")

    return dict(
        X=X, y=y, y_raw=y_raw, groups=groups, groups_arr=groups_arr,
        feature_names=feature_names, n_samples=n_samples, n_features=n_features,
        class_ratio={'neg_0': int((y == 0).sum()), 'pos_1': int((y == 1).sum())},
    )


# =============================================================================
# 2) make_model：按给定超参数构建单个 XGBoost（兼容新旧早停 API）
# =============================================================================
def make_model(params, n_estimators, seed, spw, early_stop):
    kw = dict(
        n_estimators=int(n_estimators),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        subsample=float(params['subsample']),
        colsample_bytree=float(params['colsample_bytree']),
        gamma=float(params['gamma']),
        min_child_weight=float(params['min_child_weight']),
        reg_alpha=float(params['reg_alpha']),
        reg_lambda=float(params['reg_lambda']),
        objective=FIXED_MODEL_PARAMS['objective'],
        eval_metric=FIXED_MODEL_PARAMS['eval_metric'],
        tree_method=FIXED_MODEL_PARAMS['tree_method'],
        random_state=int(seed),
        n_jobs=N_JOBS,
    )
    if USE_SCALE_POS_WEIGHT and spw is not None:
        kw['scale_pos_weight'] = float(spw)

    if early_stop:
        # 新版 XGBoost(>=1.6): early_stopping_rounds 放构造函数里
        try:
            return XGBClassifier(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **kw)
        except TypeError:
            return XGBClassifier(**kw)
    return XGBClassifier(**kw)


def fit_es(model, X_tr, y_tr, X_val, y_val):
    """带早停训练，兼容新旧 XGBoost API。"""
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        # 老版本: early_stopping_rounds 要放在 fit 里
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    return model


def predict_pos(model, X_in):
    """预测正类概率；触发过早停就用 best_iteration 截断（兼容新旧 API）。"""
    bi = best_iteration_of(model)
    if bi is not None and bi >= 0:
        try:
            return model.predict_proba(X_in, iteration_range=(0, bi + 1))[:, 1]
        except TypeError:
            pass
        try:
            return model.predict_proba(X_in, ntree_limit=bi + 1)[:, 1]
        except TypeError:
            pass
    return model.predict_proba(X_in)[:, 1]


# =============================================================================
# 3) make_group_aware_inner_split：组感知内部验证集划分（早停 + 调参共用）
#    禁止普通 train_test_split；优先 StratifiedGroupKFold，回退 GroupShuffleSplit；
#    若 y_train / y_val 出现单类则换 seed 重试，最多 INNER_MAX_RETRIES 次；仍失败 raise。
# =============================================================================
def make_group_aware_inner_split(y_tr, g_tr, base_seed, max_retries=None):
    if max_retries is None:
        max_retries = INNER_MAX_RETRIES
    y_arr = np.asarray(y_tr)
    g_arr = np.asarray(g_tr)
    idx = np.arange(len(y_arr))
    n_groups = len(np.unique(g_arr))
    if n_groups < 2:
        raise ValueError(
            f"[内部划分] 训练集分组数={n_groups} < 2，无法做 group-aware 划分。"
        )

    for k in range(max_retries):
        seed = int(base_seed) + k
        tr_idx = va_idx = None

        # 优先 StratifiedGroupKFold：兼顾“分层”和“分组不重叠”
        if HAS_SGKF:
            try:
                n_splits = max(2, min(INNER_N_SPLITS, n_groups))
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                )
                tr_idx, va_idx = next(sgkf.split(idx, y_arr, g_arr))
            except Exception:
                tr_idx = va_idx = None

        # 回退 GroupShuffleSplit（仍是 group-aware，绝非普通 train_test_split）
        if tr_idx is None:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=INNER_VALID_SIZE, random_state=seed
            )
            tr_idx, va_idx = next(gss.split(idx, y_arr, g_arr))

        # 单类检查：train / val 都必须含两类，否则换 seed 重试
        if (len(np.unique(y_arr[tr_idx])) == 2 and
                len(np.unique(y_arr[va_idx])) == 2):
            return tr_idx, va_idx

    raise ValueError(
        f"[内部划分] 在 {max_retries} 次重试后仍无法得到“双类”的 train/val 划分。"
        f"请检查该折的标签分布与分组数（n_groups={n_groups}）——很可能正类集中在"
        f"极少数 protein_id 上，导致 group-aware 划分难以两边都覆盖正类。"
    )


# =============================================================================
# 4) sample_param_grid：RandomizedSearch 风格采样（trial 0 固定为基线，去重）
# =============================================================================
def sample_param_grid(n_trials, seed):
    n_trials = max(1, int(n_trials))
    rng = np.random.RandomState(int(seed))
    grid = [dict(BASELINE_PARAMS)]                       # trial 0 = 基线兜底
    seen = {tuple(sorted(grid[0].items()))}
    attempts = 0
    cap = n_trials * 200                                 # 防止取值组合不足时死循环
    while len(grid) < n_trials and attempts < cap:
        attempts += 1
        cand = {k: _py(rng.choice(v)) for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        grid.append(cand)
    return grid


# =============================================================================
# 近常数特征筛选：只基于 outer training fold 判断（绝不基于全数据）
# =============================================================================
def select_outer_fold_features(X_tr_full, fold, outdir):
    if not DROP_NEAR_CONSTANT:
        return list(X_tr_full.columns), []
    std = X_tr_full.std(axis=0).fillna(0.0)
    nun = X_tr_full.nunique(dropna=False)
    removed = X_tr_full.columns[(std < NEAR_CONST_STD_TOL) | (nun <= 1)].tolist()
    kept = [c for c in X_tr_full.columns if c not in removed]
    if len(kept) == 0:                                   # 极端兜底：不至于删空
        kept, removed = list(X_tr_full.columns), []
    with open(os.path.join(outdir, f'fold{fold}_removed_features.txt'),
              'w', encoding='utf-8') as f:
        f.write(f"# fold{fold} 基于【外层训练折】判定的近常数/零方差列（已删除）\n")
        f.write(f"# 判定阈值: std < {NEAR_CONST_STD_TOL} 或 唯一取值数 <= 1\n")
        if removed:
            f.write("\n".join(map(str, removed)) + "\n")
        else:
            f.write("（无）\n")
    if removed:
        print(f"  [特征] fold{fold} 删除近常数列 {len(removed)} 个，剩余 {len(kept)} 个")
    return kept, removed


# =============================================================================
# 5) run_one_fold：单个外层折的完整流程
#    （外层防泄露检查 -> 特征筛选 -> 组感知内部划分 -> 随机搜索 -> 单模型重训 -> 评估）
# =============================================================================
def run_one_fold(fold, data, train_idx, test_idx, outdir, leak_log):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    X, y, y_raw = data['X'], data['y'], data['y_raw']
    groups_arr, feature_names = data['groups_arr'], data['feature_names']

    # ---- 外层 train/test 切分 ----
    X_tr_full = X.iloc[train_idx].copy()
    y_tr_full = y.iloc[train_idx].copy()
    g_tr_full = groups_arr[train_idx]
    X_te = X.iloc[test_idx].copy()
    y_te = y.iloc[test_idx].copy()
    g_te = groups_arr[test_idx]

    # ---- 防泄露检查 1：外层 train / test 的 protein_id 不能重叠 ----
    assert_no_group_overlap(g_tr_full, g_te, f"fold{fold} outer train/test", leak_log)

    if len(np.unique(y_tr_full)) < 2 or len(np.unique(y_te)) < 2:
        raise ValueError(f"[Fold {fold}] 外层 train 或 test 为单类，无法评估 AUC。")

    # ---- 近常数特征筛选：只看外层训练折 ----
    kept_cols, _removed = select_outer_fold_features(X_tr_full, fold, outdir)
    kept_idx = [feature_names.index(c) for c in kept_cols]
    X_tr_full_k = X_tr_full[kept_cols]
    X_te_k = X_te[kept_cols]

    # ---- 组感知内部划分（早停 + 调参共用同一个 val；test 折完全不参与）----
    base_seed = RANDOM_SEED + 1000 * fold
    tr_idx, va_idx = make_group_aware_inner_split(y_tr_full, g_tr_full, base_seed)
    X_train, y_train = X_tr_full_k.iloc[tr_idx], y_tr_full.iloc[tr_idx]
    X_val, y_val = X_tr_full_k.iloc[va_idx], y_tr_full.iloc[va_idx]

    # ---- 防泄露检查 2：内部 train / val 的 protein_id 不能重叠 ----
    assert_no_group_overlap(g_tr_full[tr_idx], g_tr_full[va_idx],
                            f"fold{fold} inner train/val", leak_log)

    # ---- scale_pos_weight 基准：用内部训练标签统计（不看 val、更不看 test）----
    spw_base = spw_of(y_train)

    # ---- 随机搜索：每个 trial 带早停，目标=validation AUC（绝不碰 test 折）----
    grid = sample_param_grid(N_TRIALS, RANDOM_SEED + fold)
    trials = []
    for ti, params in enumerate(grid):
        spw = params['spw_mult'] * spw_base
        # 搜索阶段固定 random_state=RANDOM_SEED，保证 trial 间可比、整体可复现
        model = make_model(params, N_ESTIMATORS, RANDOM_SEED, spw, early_stop=True)
        fit_es(model, X_train, y_train, X_val, y_val)
        bi = best_iteration_of(model)
        val_auc = roc_auc_score(y_val, predict_pos(model, X_val))
        rec = dict(params)
        rec.update(trial=ti, is_baseline=(ti == 0), spw_base=spw_base,
                   scale_pos_weight=spw, best_iteration=bi, val_auc=val_auc)
        trials.append(rec)
        if (ti + 1) % 10 == 0 or ti == 0:
            print(f"  [fold{fold}] trial {ti + 1}/{len(grid)} val_auc={val_auc:.4f}")

    trials_df = pd.DataFrame(trials).sort_values('val_auc', ascending=False) \
        .reset_index(drop=True)
    trials_df.to_csv(
        os.path.join(outdir, f'fold{fold}_xgb_hyperparam_trials.csv'), index=False
    )

    # ---- 选 best：val AUC 最高者（基线作为 trial 0 已纳入，天然兜底）----
    best_row = trials_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in SEARCH_SPACE.keys()}
    inner_val_auc = float(best_row['val_auc'])
    spw_best = float(best_row['scale_pos_weight'])
    with open(os.path.join(outdir, f'fold{fold}_best_params.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'fold': fold,
            'best_params': best_params,
            'scale_pos_weight': spw_best,
            'spw_base': spw_base,
            'inner_val_auc': inner_val_auc,
            'is_baseline': bool(best_row['is_baseline']),
            'n_kept_features': len(kept_cols),
        }, f, ensure_ascii=False, indent=2)

    # ---- 用 best_params 训练【单一】XGBoost：同一组感知 val 早停，用于 test 预测 ----
    best_model = make_model(best_params, N_ESTIMATORS, RANDOM_SEED, spw_best,
                            early_stop=True)
    fit_es(best_model, X_train, y_train, X_val, y_val)
    best_it = best_iteration_of(best_model)

    # ---- 在 test 折上评估（test 折至此才被使用，仅用于打分）----
    proba = predict_pos(best_model, X_te_k)
    pred = (proba >= 0.5).astype(int)
    metrics = dict(
        outer_test_auc=float(roc_auc_score(y_te, proba)),
        accuracy=float(accuracy_score(y_te, pred)),
        f1_score=float(f1_score(y_te, pred, zero_division=0)),
        precision=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
    )
    print(f"  Fold {fold} -> inner_val_AUC={inner_val_auc:.4f} | "
          f"outer_test_AUC={metrics['outer_test_auc']:.4f}, "
          f"ACC={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, "
          f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f} | "
          f"best_iter={best_it}, baseline={best_row['is_baseline']}")


    # ---- 保存本折预测（带 protein_id，并保持原始标签 0/1）----
    pred_df = X_te_k.copy()
    pred_df.insert(0, 'protein_id', np.asarray(g_te))
    pred_df['true_label'] = pd.Series(y_te.values).map(INVERSE_LABEL_MAP).values
    pred_df['pred_label'] = pd.Series(pred).map(INVERSE_LABEL_MAP).values
    pred_df['pred_proba'] = proba
    pred_df.to_csv(os.path.join(outdir, f'fold{fold}_xgb_predictions.csv'), index=False)

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"Fold {fold} Confusion Matrix (single XGBoost, tuned)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'fold{fold}_xgb_cm.png'), dpi=150)
    plt.close()

    return dict(
        fold=fold, test_idx=test_idx, proba=proba, pred=pred,
        best_params=best_params, scale_pos_weight=spw_best,
        inner_val_auc=inner_val_auc, best_iteration=best_it,
        n_removed_features=len(feature_names) - len(kept_cols),
        is_baseline=bool(best_row['is_baseline']),
        **metrics,
    )


# =============================================================================
# 6) run_cv：外层 GroupKFold 交叉验证编排（仅保留每折指标，不合并 OOF 预测）
# =============================================================================
def run_cv(data, outdir):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    X, y, groups = data['X'], data['y'], data['groups']

    leak_log = os.path.join(outdir, 'group_leakage_checks.txt')
    with open(leak_log, 'w', encoding='utf-8') as f:
        f.write("# 防泄露检查：所有 outer/inner 的分组(protein_id)交集都应为 0\n")

    cv = GroupKFold(n_splits=N_FOLDS)
    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups=groups), start=1):
        print(f"\n======= 第 {fold} 折 / 共 {N_FOLDS} 折 =======")
        res = run_one_fold(fold, data, train_idx, test_idx, outdir, leak_log)

        cv_results.append({
            'fold': fold,
            'outer_test_auc': res['outer_test_auc'],
            'inner_val_auc': res['inner_val_auc'],
            'accuracy': res['accuracy'],
            'f1_score': res['f1_score'],
            'precision': res['precision'],
            'recall': res['recall'],
            'best_iteration': res['best_iteration'],
            'n_removed_features': res['n_removed_features'],
            'selected_baseline': res['is_baseline'],
            'best_params': json.dumps(res['best_params'], ensure_ascii=False),
            'scale_pos_weight': res['scale_pos_weight'],
        })

    # ---- 保存 cv_xgb_results.csv（含 best_iteration / best_params / 两个 AUC）----
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(outdir, 'cv_xgb_results.csv'), index=False)

    if len(cv_df) > 0:
        print("\n========== 交叉验证汇总 (mean ± std) ==========")
        for col in ['outer_test_auc', 'inner_val_auc', 'accuracy',
                    'f1_score', 'precision', 'recall']:
            print(f"{col:>16}: {cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}")

    return cv_df


# =============================================================================
# 7) train_final_model：全数据训练最终单模型（仅用于部署，不用于报告 CV 指标）
#    用“inner_val_auc 最高的那一折”的 best_params；spw 按全数据训练标签 × 同样的 mult 重算。
# =============================================================================
def train_final_model(data, cv_df, outdir):
    """训练最终模型并保存 xgb_final_model.joblib。

    流程：
    1) 从交叉验证结果中选择 inner_val_auc 最高那一折的 best_params；
    2) 在全数据上再做一次 group-aware train/val，仅用于估计 best_iteration；
    3) 使用 best_iteration + 1 作为树数，关闭早停，在全部 X/y 上重新训练最终模型。

    因此，最终保存的模型是真正用全部样本 fit 的模型；CV 指标仍只来自各折 outer test。
    """
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    if len(cv_df) == 0:
        print("\n[最终模型] 没有有效折，跳过。")
        return None

    best_fold_row = cv_df.sort_values('inner_val_auc', ascending=False).iloc[0]
    best_params = json.loads(best_fold_row['best_params'])
    print(f"\n===== 训练最终全量模型（取 fold{int(best_fold_row['fold'])} "
          f"的 best_params，inner_val_auc={best_fold_row['inner_val_auc']:.4f}）=====")

    X, y, groups_arr = data['X'], data['y'], data['groups_arr']

    # 先用全数据上的一次 group-aware 内部划分估计 best_iteration（只用于确定树数）
    tr_idx, va_idx = make_group_aware_inner_split(y, groups_arr, RANDOM_SEED + 99999)
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    spw_es = best_params['spw_mult'] * spw_of(y_tr)
    es_model = make_model(best_params, N_ESTIMATORS, RANDOM_SEED, spw_es, early_stop=True)
    fit_es(es_model, X_tr, y_tr, X_va, y_va)
    best_it = best_iteration_of(es_model)
    final_n_estimators = int(best_it + 1) if best_it is not None and best_it >= 0 else N_ESTIMATORS

    # 真正最终模型：关闭 early stopping，在全部数据上重新 fit
    spw_final = best_params['spw_mult'] * spw_of(y)
    final_model = make_model(best_params, final_n_estimators, RANDOM_SEED, spw_final,
                             early_stop=False)
    final_model.fit(X, y, verbose=False)

    model_path = os.path.join(outdir, 'xgb_final_model.joblib')
    joblib.dump({
        'model': final_model,
        'features': data['feature_names'],
        'label_map': LABEL_MAP,
        'inverse_label_map': INVERSE_LABEL_MAP,
        'best_params': best_params,
        'scale_pos_weight': spw_final,
        'final_n_estimators': final_n_estimators,
        'early_stop_best_iteration_reference': best_it,
        'from_fold': int(best_fold_row['fold']),
        'note': '最终部署/SHAP 用模型：先用 group-aware 内部验证估计 best_iteration，'
                '再关闭早停并用全部 X/y 重新训练；CV 指标必须使用 cv_xgb_results.csv。',
    }, model_path)

    with open(os.path.join(outdir, 'final_best_params.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_params,
            'scale_pos_weight': spw_final,
            'final_n_estimators': final_n_estimators,
            'early_stop_best_iteration_reference': best_it,
            'from_fold': int(best_fold_row['fold']),
            'trained_on_all_samples': True,
            'deployment_only': True,
        }, f, ensure_ascii=False, indent=2)

    print(f"最终全量模型已保存: {model_path}  （final_n_estimators={final_n_estimators}）")
    return final_model

# =============================================================================
# 8) save_summary：写出 summary.json（均值/方差、全部配置、数据维度、类别比例、时间戳）
# =============================================================================
def save_summary(data, cv_df, outdir, elapsed_sec):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    def _ms(col):
        return (float(cv_df[col].mean()), float(cv_df[col].std())) \
            if len(cv_df) else (float('nan'), float('nan'))

    mean_auc, std_auc = _ms('outer_test_auc')
    mean_f1, std_f1 = _ms('f1_score')
    mean_acc, std_acc = _ms('accuracy')
    mean_prec, std_prec = _ms('precision')
    mean_rec, std_rec = _ms('recall')

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'single XGBoost (group-aware nested random search, fold-wise outer-test predictions)',
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'mean_accuracy': mean_acc, 'std_accuracy': std_acc,
        'mean_precision': mean_prec, 'std_precision': std_prec,
        'mean_recall': mean_rec, 'std_recall': std_rec,
        'baseline_auc': BASELINE_AUC,
        'auc_minus_baseline': (mean_auc - BASELINE_AUC) if not np.isnan(mean_auc) else None,
        'n_folds_evaluated': int(len(cv_df)),
        'data': {
            'data_path': DATA_REFERENCE,
            'n_samples': int(data['n_samples']),
            'n_features': int(data['n_features']),
            'n_groups': int(len(np.unique(data['groups_arr']))),
            'class_ratio_0_1': data['class_ratio'],
        },
        'config': {
            'RANDOM_SEED': RANDOM_SEED, 'N_FOLDS': N_FOLDS, 'N_TRIALS': N_TRIALS,
            'N_ESTIMATORS': N_ESTIMATORS,
            'EARLY_STOPPING_ROUNDS': EARLY_STOPPING_ROUNDS,
            'INNER_N_SPLITS': INNER_N_SPLITS,
            'INNER_VALID_SIZE': INNER_VALID_SIZE,
            'INNER_MAX_RETRIES': INNER_MAX_RETRIES,
            'USE_SCALE_POS_WEIGHT': USE_SCALE_POS_WEIGHT,
            'DROP_NEAR_CONSTANT': DROP_NEAR_CONSTANT,
            'NEAR_CONST_STD_TOL': NEAR_CONST_STD_TOL,
            'inner_split_method': 'StratifiedGroupKFold' if HAS_SGKF else 'GroupShuffleSplit',
            'search_space': SEARCH_SPACE,
            'baseline_params': BASELINE_PARAMS,
        },
        'per_fold': cv_df.to_dict(orient='records') if len(cv_df) else [],
        'elapsed_sec': round(float(elapsed_sec), 2),
    }
    with open(os.path.join(outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# =============================================================================
# 主流程
# =============================================================================
def run_analysis():
    global OUTDIR
    t0 = time.time()

    # 固定输出到当前脚本所在目录下的 fin_01_xgboost_remove_resn 文件夹
    OUTDIR = os.path.abspath(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 68)
    print("单一 XGBoost + 组感知调参（保留逐折 outer-test 预测）")
    print(f"  外层折数 N_FOLDS = {N_FOLDS} | 每折随机搜索 N_TRIALS = {N_TRIALS}")
    print(f"  内部划分 = {'StratifiedGroupKFold' if HAS_SGKF else 'GroupShuffleSplit'}"
          f"（group-aware；禁止普通 train_test_split）")
    print(f"  合并 OOF/SHAP = 未执行 | 输出目录 = {OUTDIR}")
    print("=" * 68)

    # 1) 数据
    data = load_data(DATA_PATH)

    # 2) 外层 GroupKFold 交叉验证（每折内部：组感知划分 + 随机搜索 + 单模型）
    cv_df = run_cv(data, OUTDIR)

    # 3) 最终全量模型（部署/后续 SHAP 用；不用于 CV 指标）
    train_final_model(data, cv_df, OUTDIR)

    # 4) summary.json
    summary = save_summary(data, cv_df, OUTDIR, time.time() - t0)

    # ---- 结束语（按要求打印固定三行）----
    mean_auc = summary['mean_auc']
    std_auc = summary['std_auc']
    print("\n" + "=" * 68)
    if not np.isnan(mean_auc):
        delta = mean_auc - BASELINE_AUC
        print(f"对比基线 {BASELINE_AUC:.4f}: {'+' if delta >= 0 else ''}{delta:.4f}")
    print("Group-aware tuned XGBoost finished.")
    print(f"Mean outer GroupKFold AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Results saved to: {OUTDIR}")
    print("=" * 68)
    return summary


def main(argv=None):
    return run_ablation_cli(__file__, "remove_contacts", globals(), run_analysis, argv)


if __name__ == '__main__':
    main()
