#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1 xgboost: protein-cluster bootstrap of saved outer-test predictions.

Resampling, classification rules, fold averaging and percentile statistics are
retained from the supplied script. This analysis does not refit the models.
Run from the repository root:
    python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py --result-dir PATH

Default settings: configs/task1/evaluation/bootstrap/xgboost.json.
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"protein_id", "true_label", "pred_proba"}
METRIC_NAMES = (
    "roc_auc",
    "accuracy",
    "f1_binary_label1",
    "f1_macro",
    "precision_label1",
    "recall_label1",
)


def parse_args(argv=None) -> argparse.Namespace:
    args = bootstrap_args(__file__, "xgboost", argv)
    return args


def discover_prediction_files(result_dir: Path) -> List[Tuple[int, Path]]:
    """兼容 fold1_xgb_predictions.csv 与 fold_1_xgb_predictions.csv。"""
    candidates = list(result_dir.glob("fold*xgb_predictions.csv"))
    pattern = re.compile(r"fold_?(\d+)_xgb_predictions\.csv$", re.IGNORECASE)
    found: List[Tuple[int, Path]] = []

    for path in candidates:
        match = pattern.search(path.name)
        if match:
            found.append((int(match.group(1)), path))

    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(
            f"在 {result_dir.resolve()} 中未找到 fold1_xgb_predictions.csv 等文件。"
        )

    fold_ids = [fold for fold, _ in found]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"检测到重复 fold 编号：{fold_ids}")
    return found


def _read_prediction_columns(path: Path) -> pd.DataFrame:
    """只读取统计所需列，避免载入预测文件中附带的数百个特征列。"""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(header)
    if missing:
        raise ValueError(f"{path.name} 缺少列：{sorted(missing)}")

    use_cols = ["protein_id", "true_label", "pred_proba"]
    if "pred_label" in header:
        use_cols.append("pred_label")
    return pd.read_csv(path, usecols=use_cols)


def validate_binary_0_1(series: pd.Series, column_name: str) -> pd.Series:
    """标签本身就是 0/1；仅做数值与取值范围检查，不再映射。"""
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any():
        raise ValueError(f"{column_name} 中存在缺失值。")
    values = values.astype(int)
    unique = set(values.unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"{column_name} 应为原生标签 0/1，实际取值：{sorted(unique)}"
        )
    return values.astype(np.int8)


def load_predictions(result_dir: Path) -> pd.DataFrame:
    files = discover_prediction_files(result_dir)
    frames: List[pd.DataFrame] = []

    for fold, path in files:
        part = _read_prediction_columns(path)
        part.insert(0, "fold", fold)
        part["source_file"] = path.name
        frames.append(part)

    all_pred = pd.concat(frames, ignore_index=True)

    if all_pred["protein_id"].isna().any():
        raise ValueError("protein_id 中存在缺失值，无法执行 protein-cluster Bootstrap。")

    all_pred["y_true"] = validate_binary_0_1(
        all_pred["true_label"], "true_label"
    )
    all_pred["pred_proba"] = pd.to_numeric(
        all_pred["pred_proba"], errors="coerce"
    )
    if all_pred["pred_proba"].isna().any():
        raise ValueError("pred_proba 中存在 NaN 或非数值。")
    if ((all_pred["pred_proba"] < 0) | (all_pred["pred_proba"] > 1)).any():
        raise ValueError("pred_proba 必须位于 [0, 1]。")

    # 与训练脚本一致：标签 1 为正类，pred_proba 的分类阈值为 0.5。
    all_pred["y_pred"] = (all_pred["pred_proba"] >= 0.5).astype(np.int8)

    # pred_label 本身就是 0/1；只检查其与 pred_proba>=0.5 是否一致。
    if "pred_label" in all_pred.columns and all_pred["pred_label"].notna().all():
        saved_pred = validate_binary_0_1(all_pred["pred_label"], "pred_label")
        mismatch = int((saved_pred.to_numpy() != all_pred["y_pred"].to_numpy()).sum())
        if mismatch:
            print(
                f"[警告] 保存的 pred_label 与 pred_proba>=0.5 有 {mismatch} 行不一致；"
                "后续统一以 pred_proba>=0.5 为准。",
                file=sys.stderr,
            )
        else:
            print("pred_label 与 pred_proba>=0.5 完全一致。")

    # 同一 protein_id 必须只属于一个 outer-test fold；标签可在蛋白内部变化。
    fold_per_group = all_pred.groupby("protein_id", dropna=False)["fold"].nunique()
    bad_fold_groups = fold_per_group[fold_per_group > 1]
    if not bad_fold_groups.empty:
        raise ValueError(
            "同一 protein_id 出现在多个 outer-test fold 中，疑似交叉验证分组泄露。"
            f"示例：{bad_fold_groups.index[:10].tolist()}"
        )

    if all_pred["y_true"].nunique() != 2:
        raise ValueError("合并后的 outer-test 预测不包含两个类别，无法计算二分类 AUC。")

    return all_pred


def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def metrics_from_confusion_and_auc(
    tn: float,
    fp: float,
    fn: float,
    tp: float,
    auc: float,
) -> Dict[str, float]:
    total = tn + fp + fn + tp
    f1_pos = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    f1_neg = _safe_divide(2.0 * tn, 2.0 * tn + fp + fn)
    return {
        "roc_auc": float(auc),
        "accuracy": _safe_divide(tp + tn, total),
        "f1_binary_label1": f1_pos,
        "f1_macro": 0.5 * (f1_pos + f1_neg),
        "precision_label1": _safe_divide(tp, tp + fp),
        "recall_label1": _safe_divide(tp, tp + fn),
    }


@dataclass
class FoldStats:
    fold: int
    group_ids: np.ndarray
    n_rows: np.ndarray
    pos_count: np.ndarray
    neg_count: np.ndarray
    tp_count: np.ndarray
    fp_count: np.ndarray
    tn_count: np.ndarray
    fn_count: np.ndarray
    auc_pair_contribution: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(len(self.group_ids))

    def evaluate_multiplicity(self, multiplicity: np.ndarray) -> Dict[str, float]:
        m = np.asarray(multiplicity, dtype=float)
        if m.shape != (self.n_groups,):
            raise ValueError(
                f"fold {self.fold}: multiplicity 形状应为 {(self.n_groups,)}，"
                f"实际为 {m.shape}"
            )

        n_pos = float(m @ self.pos_count)
        n_neg = float(m @ self.neg_count)
        if n_pos <= 0 or n_neg <= 0:
            raise ValueError("Bootstrap 样本缺少一个真实类别，AUC 无法计算。")

        auc_num = float(np.einsum("i,ij,j->", m, self.auc_pair_contribution, m))
        auc = auc_num / (n_pos * n_neg)

        tp = float(m @ self.tp_count)
        fp = float(m @ self.fp_count)
        tn = float(m @ self.tn_count)
        fn = float(m @ self.fn_count)
        return metrics_from_confusion_and_auc(tn=tn, fp=fp, fn=fn, tp=tp, auc=auc)


def build_fold_stats(df_fold: pd.DataFrame) -> FoldStats:
    fold = int(df_fold["fold"].iloc[0])
    group_ids = pd.unique(df_fold["protein_id"])
    n_groups = len(group_ids)
    group_to_index = {group_id: i for i, group_id in enumerate(group_ids)}

    n_rows = np.zeros(n_groups, dtype=np.int64)
    pos_count = np.zeros(n_groups, dtype=np.int64)
    neg_count = np.zeros(n_groups, dtype=np.int64)
    tp_count = np.zeros(n_groups, dtype=np.int64)
    fp_count = np.zeros(n_groups, dtype=np.int64)
    tn_count = np.zeros(n_groups, dtype=np.int64)
    fn_count = np.zeros(n_groups, dtype=np.int64)

    positive_scores: List[np.ndarray] = []
    negative_scores_sorted: List[np.ndarray] = []

    for group_id in group_ids:
        idx = group_to_index[group_id]
        block = df_fold.loc[df_fold["protein_id"] == group_id]
        y = block["y_true"].to_numpy(dtype=np.int8)
        pred = block["y_pred"].to_numpy(dtype=np.int8)
        scores = block["pred_proba"].to_numpy(dtype=float)

        n_rows[idx] = len(block)
        pos_count[idx] = int((y == 1).sum())
        neg_count[idx] = int((y == 0).sum())
        tp_count[idx] = int(((y == 1) & (pred == 1)).sum())
        fp_count[idx] = int(((y == 0) & (pred == 1)).sum())
        tn_count[idx] = int(((y == 0) & (pred == 0)).sum())
        fn_count[idx] = int(((y == 1) & (pred == 0)).sum())

        positive_scores.append(scores[y == 1])
        negative_scores_sorted.append(np.sort(scores[y == 0]))

    # A[i, j] = protein i 中所有正例与 protein j 中所有负例的胜出数 + 0.5*平局数。
    auc_pair = np.zeros((n_groups, n_groups), dtype=np.float64)
    for i, pos_scores in enumerate(positive_scores):
        if pos_scores.size == 0:
            continue
        for j, neg_scores in enumerate(negative_scores_sorted):
            if neg_scores.size == 0:
                continue
            lower = np.searchsorted(neg_scores, pos_scores, side="left")
            upper = np.searchsorted(neg_scores, pos_scores, side="right")
            auc_pair[i, j] = float(
                np.sum(lower + 0.5 * (upper - lower), dtype=np.float64)
            )

    stats = FoldStats(
        fold=fold,
        group_ids=np.asarray(group_ids),
        n_rows=n_rows,
        pos_count=pos_count,
        neg_count=neg_count,
        tp_count=tp_count,
        fp_count=fp_count,
        tn_count=tn_count,
        fn_count=fn_count,
        auc_pair_contribution=auc_pair,
    )

    # 自检：所有蛋白重数=1 时，应恢复该折原始逐帧指标。
    original = stats.evaluate_multiplicity(np.ones(n_groups, dtype=int))
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = df_fold["y_true"].to_numpy(dtype=int)
        y_pred = df_fold["y_pred"].to_numpy(dtype=int)
        y_prob = df_fold["pred_proba"].to_numpy(dtype=float)
        reference = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_binary_label1": float(f1_score(y_true, y_pred, zero_division=0)),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "precision_label1": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall_label1": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        max_diff = max(abs(original[k] - reference[k]) for k in METRIC_NAMES)
        if max_diff > 1e-10:
            raise RuntimeError(
                f"fold {fold}: 预计算指标与 sklearn 参考值不一致，最大差异={max_diff:.3g}"
            )
    except ImportError:
        pass

    return stats


def build_all_fold_stats(all_pred: pd.DataFrame) -> Dict[int, FoldStats]:
    fold_stats: Dict[int, FoldStats] = {}
    for fold, df_fold in all_pred.groupby("fold", sort=True):
        print(
            f"预计算 fold {int(fold)}：{df_fold['protein_id'].nunique()} 个 protein_id，"
            f"{len(df_fold)} 帧……"
        )
        fold_stats[int(fold)] = build_fold_stats(df_fold)
    return fold_stats


def recompute_fold_metrics(fold_stats: Dict[int, FoldStats]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for fold in sorted(fold_stats):
        stats = fold_stats[fold]
        rec: Dict[str, float] = {
            "fold": int(fold),
            "n_rows": int(stats.n_rows.sum()),
            "n_proteins": int(stats.n_groups),
        }
        rec.update(stats.evaluate_multiplicity(np.ones(stats.n_groups, dtype=int)))
        records.append(rec)
    return pd.DataFrame(records)


def compare_with_existing_cv(result_dir: Path, fold_metrics: pd.DataFrame) -> None:
    cv_path = result_dir / "cv_xgb_results.csv"
    if not cv_path.exists():
        print("[提示] 未找到 cv_xgb_results.csv，跳过一致性核对。", file=sys.stderr)
        return

    cv = pd.read_csv(cv_path)
    required = {"fold", "outer_test_auc", "accuracy", "f1_score", "precision", "recall"}
    if not required.issubset(cv.columns):
        print(
            "[提示] cv_xgb_results.csv 列名不完整，跳过一致性核对。",
            file=sys.stderr,
        )
        return

    check = fold_metrics.rename(
        columns={
            "roc_auc": "recomputed_roc_auc",
            "accuracy": "recomputed_accuracy",
            "f1_binary_label1": "recomputed_f1",
            "precision_label1": "recomputed_precision",
            "recall_label1": "recomputed_recall",
        }
    )
    merged = cv.merge(check, on="fold", how="inner", validate="one_to_one")
    pairs = {
        "outer_test_auc": "recomputed_roc_auc",
        "accuracy": "recomputed_accuracy",
        "f1_score": "recomputed_f1",
        "precision": "recomputed_precision",
        "recall": "recomputed_recall",
    }
    max_diff = 0.0
    for old_col, new_col in pairs.items():
        diff = np.max(
            np.abs(
                merged[old_col].to_numpy(dtype=float)
                - merged[new_col].to_numpy(dtype=float)
            )
        )
        max_diff = max(max_diff, float(diff))

    if max_diff > 1e-8:
        print(
            f"[警告] 预测文件重算指标与 cv_xgb_results.csv 最大差异={max_diff:.6g}；"
            "请确认二者来自同一次训练。",
            file=sys.stderr,
        )
    else:
        print("预测文件重算指标与 cv_xgb_results.csv 一致。")


def draw_valid_multiplicity(
    stats: FoldStats,
    rng: np.random.Generator,
    max_redraw: int,
) -> Tuple[np.ndarray, int]:
    """普通 protein-cluster bootstrap；若抽样后缺少一类，则重抽。"""
    p = np.full(stats.n_groups, 1.0 / stats.n_groups, dtype=float)
    for redraw in range(max_redraw + 1):
        multiplicity = rng.multinomial(stats.n_groups, p)
        n_pos = float(multiplicity @ stats.pos_count)
        n_neg = float(multiplicity @ stats.neg_count)
        if n_pos > 0 and n_neg > 0:
            return multiplicity, redraw
    raise RuntimeError(
        f"fold {stats.fold}: 连续 {max_redraw} 次重抽后仍无法同时包含两类。"
        "请检查该 fold 中两类是否集中在极少数 protein_id。"
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
    folds = sorted(fold_stats)

    for b in range(n_bootstrap):
        per_fold: List[Dict[str, float]] = []
        for fold in folds:
            stats = fold_stats[fold]
            multiplicity, redraws = draw_valid_multiplicity(
                stats=stats, rng=rng, max_redraw=max_redraw
            )
            total_redraws += redraws
            per_fold.append(stats.evaluate_multiplicity(multiplicity))

        rec: Dict[str, float] = {"bootstrap_id": b + 1}
        for metric in METRIC_NAMES:
            rec[metric] = float(np.mean([item[metric] for item in per_fold]))
        records.append(rec)

        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"Bootstrap: {b + 1}/{n_bootstrap}")

    return pd.DataFrame(records), total_redraws


def summarize_ci(
    fold_metrics: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    alpha = 1.0 - confidence
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    records: List[Dict[str, float]] = []
    for metric in METRIC_NAMES:
        point = float(fold_metrics[metric].mean())
        values = bootstrap_df[metric].to_numpy(dtype=float)
        records.append(
            {
                "metric": metric,
                "point_estimate_mean_outer_folds": point,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_level": float(confidence),
                "ci_lower_percentile": float(np.percentile(values, lower_q)),
                "ci_upper_percentile": float(np.percentile(values, upper_q)),
                "n_bootstrap": int(len(values)),
            }
        )
    return pd.DataFrame(records)


def protein_label_composition(all_pred: pd.DataFrame) -> pd.DataFrame:
    grouped = all_pred.groupby(["fold", "protein_id"], sort=True, dropna=False)
    rows: List[Dict[str, float]] = []
    for (fold, protein_id), block in grouped:
        n_label0 = int((block["y_true"] == 0).sum())
        n_label1 = int((block["y_true"] == 1).sum())
        total = n_label0 + n_label1
        rows.append(
            {
                "fold": int(fold),
                "protein_id": protein_id,
                "n_frames": int(total),
                "n_true_label_0": n_label0,
                "n_true_label_1": n_label1,
                "fraction_true_label_1": float(n_label1 / total) if total else np.nan,
                "contains_both_labels": bool(n_label0 > 0 and n_label1 > 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(args.effective_config, ensure_ascii=False, indent=2))
        return
    result_dir = args.result_dir.expanduser().resolve()

    if args.n_bootstrap < 100:
        raise ValueError("n_bootstrap 过少；建议至少 500。")
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("confidence 必须位于 (0, 1)。")
    if args.max_redraw < 0:
        raise ValueError("max_redraw 不能为负数。")

    # 输入结果仍从 result_dir 读取；所有 Bootstrap 输出统一放入 bootstrap 子目录。
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入结果目录: {result_dir}")
    print(f"Bootstrap 输出目录: {output_dir}")
    all_pred = load_predictions(result_dir)

    composition = protein_label_composition(all_pred)
    frame_counts = composition["n_frames"]
    n_mixed = int(composition["contains_both_labels"].sum())
    print(
        "合并后的 outer-test 预测："
        f"{len(all_pred)} 帧，{all_pred['protein_id'].nunique()} 个 protein_id，"
        f"{all_pred['fold'].nunique()} 个 folds。"
    )
    print(
        "每个 protein_id 的帧数："
        f"min={int(frame_counts.min())}, median={float(frame_counts.median()):.1f}, "
        f"max={int(frame_counts.max())}。"
    )
    print(
        f"同一 protein_id 内同时包含标签 0 和 1：{n_mixed}/{len(composition)} 个。"
    )
    print(
        "Bootstrap 单位：protein_id；不按蛋白标签分层，因为标签是帧级且可在同一蛋白内变化。"
    )

    fold_stats = build_all_fold_stats(all_pred)
    fold_metrics = recompute_fold_metrics(fold_stats)
    compare_with_existing_cv(result_dir, fold_metrics)

    print("\n原始 5 折均值：")
    for metric in METRIC_NAMES:
        print(f"  {metric:>22s} = {fold_metrics[metric].mean():.6f}")

    bootstrap_df, total_redraws = bootstrap_fold_mean(
        fold_stats=fold_stats,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        max_redraw=args.max_redraw,
    )
    summary = summarize_ci(
        fold_metrics=fold_metrics,
        bootstrap_df=bootstrap_df,
        confidence=args.confidence,
    )

    fold_out = output_dir / "recomputed_fold_metrics.csv"
    dist_out = output_dir / "bootstrap_distribution.csv"
    summary_out = output_dir / "bootstrap_ci_summary.csv"
    composition_out = output_dir / "protein_label_composition.csv"
    metadata_out = output_dir / "bootstrap_metadata.json"

    fold_metrics.to_csv(fold_out, index=False)
    bootstrap_df.to_csv(dist_out, index=False)
    summary.to_csv(summary_out, index=False)
    composition.to_csv(composition_out, index=False)

    metadata = {
        "result_dir": public_path(result_dir),
        "output_dir": public_path(output_dir),
        "prediction_files": sorted(all_pred["source_file"].unique().tolist()),
        "n_rows_frames": int(len(all_pred)),
        "n_proteins": int(all_pred["protein_id"].nunique()),
        "n_folds": int(all_pred["fold"].nunique()),
        "n_proteins_with_both_labels": n_mixed,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "confidence": float(args.confidence),
        "bootstrap_unit": "protein_id cluster within each outer fold",
        "within_protein_labels": "may vary by frame; no group-level label stratification",
        "point_estimate": "mean of the outer-test fold metrics",
        "ci_method": "percentile cluster bootstrap",
        "positive_class": "label 1",
        "prediction_threshold": 0.5,
        "total_redraws_due_to_single_class_bootstrap_samples": int(total_redraws),
        "implementation": (
            "Exact cluster-multiplicity metrics using precomputed confusion counts and "
            "protein-pair AUC contributions; equivalent to duplicating all frames of each "
            "resampled protein_id."
        ),
        "interpretation": (
            "Post-hoc uncertainty of fixed outer-fold predictions under resampling of "
            "protein systems. It accounts for within-protein frame dependence at the "
            "cluster level, but does not include full model retraining or hyperparameter-"
            "search uncertainty."
        ),
    }
    with metadata_out.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nBootstrap 置信区间：")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.metric:>22s}: {row.point_estimate_mean_outer_folds:.6f} "
            f"[{row.ci_lower_percentile:.6f}, {row.ci_upper_percentile:.6f}]"
        )

    print(f"\n所有 Bootstrap 输出均已保存到：{output_dir}")
    print("已保存：")
    print(f"  {summary_out}")
    print(f"  {dist_out}")
    print(f"  {fold_out}")
    print(f"  {composition_out}")
    print(f"  {metadata_out}")

    inputs = [path for _, path in discover_prediction_files(result_dir)]
    cv_path = result_dir / "cv_xgb_results.csv"
    if cv_path.is_file():
        inputs.append(cv_path)
    save_run_info(output_dir, args, __file__, inputs)


if __name__ == "__main__":
    main()
