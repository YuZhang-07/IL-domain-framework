#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task1: refit the final XGBoost model for a subsequent SHAP analysis.

This entry point preserves the supplied model-refit protocol. It reads existing
per-fold best_params, compares their unique candidates with full-data internal
group-aware CV, takes the selected candidate's median early-stopped tree count,
and fits one final model on all supplied samples without early stopping.

It does not compute SHAP values or plots and does not rerun outer-test AUC.
See docs/shap_model_refit/README.md for inputs, outputs and limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
import platform
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    StratifiedGroupKFold = None  # type: ignore[assignment]
    HAS_SGKF = False


# =============================================================================
# 固定规则：与原训练脚本保持一致
# =============================================================================
RANDOM_SEED = 42
INNER_N_SPLITS = 5
N_ESTIMATORS_MAX = 3000
EARLY_STOPPING_ROUNDS = 100
N_JOBS = -1
LABEL_MAP = {0: 0, 1: 1}
INVERSE_LABEL_MAP = {0: 0, 1: 1}

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
TASK = "task1"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / TASK / "evaluation" / "shap" / "refit_xgboost.json"


# =============================================================================
# 路径与通用工具
# =============================================================================
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_reference(path: Path) -> str:
    """Repository-relative reference, without external directory prefixes.

    This only masks directories in explicitly recorded path fields. It does not
    anonymize scientific feature names, file basenames, or Python tracebacks.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def resolve_input_path(value: str, base: Path, label: str) -> Path:
    """Resolve one explicit/configured file; never search other runs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: a non-empty file path is required.")
    path = Path(value).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path_reference(path)}")
    return path


def load_cli_config(argv: Sequence[str] | None = None) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Read JSON settings and command-line overrides without changing the protocol."""
    parser = argparse.ArgumentParser(
        description=(
            f"{TASK}: refit an XGBoost model for later SHAP analysis. "
            "Does NOT compute SHAP values or rerun outer-test evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON configuration; an explicit relative path is relative to cwd.")
    parser.add_argument("--data", help="Feature CSV; first column=group, last column=raw label.")
    parser.add_argument("--cv-results", help="Existing cv_xgb_results.csv from the same task/run.")
    parser.add_argument("--output-dir", "--outdir", dest="output_dir", type=Path,
                        help="New or empty output directory; --outdir is a legacy alias.")
    parser.add_argument("--n-splits", type=int, help="Internal group-aware folds; default from JSON.")
    parser.add_argument("--seed", type=int, help="Random seed; default from JSON.")
    parser.add_argument("--n-jobs", type=int, help="XGBoost threads; -1 retains the source default.")
    parser.add_argument("--n-estimators-max", type=int, help="Tree cap during internal selection.")
    parser.add_argument("--early-stopping-rounds", type=int, help="Internal-validation patience.")
    parser.add_argument("--max-split-retries", type=int, help="Seeds attempted for valid splits.")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {path_reference(config_path)}")
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        parser.error("Configuration must be a JSON object with schema_version=1.")
    if config.get("task") != TASK:
        parser.error(f"This entry point requires a {TASK} configuration.")
    if config.get("stage") != "final_model_refit_for_shap":
        parser.error("Incorrect configuration stage; expected final_model_refit_for_shap.")
    paths = config.get("paths")
    settings = config.get("refit")
    if not isinstance(paths, dict) or not isinstance(settings, dict):
        parser.error("Configuration must contain paths and refit objects.")
    settings = dict(settings)
    overrides = {
        "n_splits": args.n_splits,
        "random_seed": args.seed,
        "n_jobs": args.n_jobs,
        "n_estimators_max": args.n_estimators_max,
        "early_stopping_rounds": args.early_stopping_rounds,
        "max_split_retries": args.max_split_retries,
    }
    for name, override in overrides.items():
        if override is not None:
            settings[name] = override
    expected = set(overrides)
    if set(settings) != expected:
        parser.error(f"refit must contain exactly: {', '.join(sorted(expected))}")
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int):
            parser.error(f"refit.{name} must be an integer.")
    if settings["n_splits"] < 2:
        parser.error("n_splits must be at least 2.")
    if not (0 <= settings["random_seed"] < 2**32):
        parser.error("random_seed must be in [0, 2**32).")
    if settings["n_jobs"] != -1 and settings["n_jobs"] < 1:
        parser.error("n_jobs must be -1 or a positive integer.")
    for name in ("n_estimators_max", "early_stopping_rounds", "max_split_retries"):
        if settings[name] < 1:
            parser.error(f"{name} must be positive.")

    cv_value = args.cv_results if args.cv_results is not None else paths.get("cv_results")
    if cv_value is None:
        parser.error("Pass --cv-results for a specific run, or set paths.cv_results in the JSON.")
    args.cv_path = resolve_input_path(
        cv_value, Path.cwd() if args.cv_results is not None else REPO_ROOT, "Existing CV CSV"
    )
    args.data_path = resolve_input_path(
        args.data if args.data is not None else paths.get("data"),
        Path.cwd() if args.data is not None else REPO_ROOT, "Feature CSV"
    )
    if args.output_dir is None:
        root_value = paths.get("output_root")
        if not isinstance(root_value, str) or not root_value.strip():
            parser.error("paths.output_root must be a non-empty path string.")
        root_path = Path(root_value).expanduser()
        if not root_path.is_absolute():
            root_path = REPO_ROOT / root_path
        args.outdir = (root_path / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    else:
        args.outdir = args.output_dir.expanduser().resolve()
    args.config_path = config_path
    return args, settings


def prepare_output_directory(outdir: Path) -> None:
    """Refuse occupied output directories before any expensive fitting."""
    if outdir.exists():
        if not outdir.is_dir():
            raise FileExistsError(f"Output is not a directory: {path_reference(outdir)}")
        if any(outdir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {path_reference(outdir)}. "
                "Choose a new --output-dir; existing results will not be overwritten."
            )
    else:
        outdir.mkdir(parents=True, exist_ok=False)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_run_context(args: argparse.Namespace, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Record only relevant settings/versions; never dump environment variables."""
    effective = {
        "schema_version": 1, "task": TASK, "stage": "final_model_refit_for_shap",
        "source_config": path_reference(args.config_path),
        "paths": {
            "data": path_reference(args.data_path),
            "cv_results": path_reference(args.cv_path),
            "output_dir": path_reference(args.outdir),
        },
        "refit": settings,
        "label_map": LABEL_MAP,
        "inverse_label_map": INVERSE_LABEL_MAP,
        "path_note": "External directory prefixes are masked; supply local paths when rerunning.",
    }
    write_json(args.outdir / "run_config.json", effective)
    versions = {}
    for name in ("numpy", "pandas", "xgboost", "scikit-learn", "scipy", "joblib", "threadpoolctl"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    write_json(args.outdir / "environment.json", {
        "python": platform.python_version(), "packages": versions,
        "note": "Environment of this run, not necessarily the original manuscript environment.",
    })
    return {
        "task": TASK, "stage": "final_model_refit_for_shap", "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": path_reference(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config_path),
        "data": path_reference(args.data_path), "data_sha256": sha256_file(args.data_path),
        "cv_results": path_reference(args.cv_path),
        "cv_results_sha256_before": sha256_file(args.cv_path),
        "outer_test_predictions_recomputed": False, "shap_values_computed": False,
    }



def parse_params(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        params = dict(value)
    else:
        text = str(value).strip()
        try:
            params = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 best_params：{text[:200]}") from exc

    required = {
        "max_depth", "learning_rate", "min_child_weight", "gamma",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "spw_mult",
    }
    missing = sorted(required.difference(params))
    if missing:
        raise ValueError(f"best_params 缺少字段：{missing}")

    return {
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "min_child_weight": float(params["min_child_weight"]),
        "gamma": float(params["gamma"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "spw_mult": float(params["spw_mult"]),
    }


def canonical_param_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spw_of(y: Sequence[int] | pd.Series | np.ndarray) -> float:
    arr = np.asarray(y, dtype=int)
    pos = int((arr == 1).sum())
    neg = int((arr == 0).sum())
    if pos == 0:
        raise ValueError("训练标签中没有正类1，无法计算 scale_pos_weight。")
    return float(neg / pos)


def best_iteration_of(model: XGBClassifier) -> int | None:
    value = getattr(model, "best_iteration", None)
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return int(value)


# =============================================================================
# 数据和既有 CV 结果
# =============================================================================
def load_data(data_path: Path) -> Dict[str, Any]:
    print(f"[数据] 读取：{path_reference(data_path)}")
    df = pd.read_csv(data_path)
    if df.shape[1] < 3:
        raise ValueError("输入 CSV 至少应包含 protein_id、一个特征和标签三类列。")

    group_col = str(df.columns[0])
    label_col = str(df.columns[-1])
    feature_names = [str(c) for c in df.columns[1:-1]]

    groups = df.iloc[:, 0].to_numpy()
    y_raw = pd.to_numeric(df.iloc[:, -1], errors="raise").astype(int)
    unique_labels = sorted(pd.unique(y_raw).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"标签必须已经是0/1，实际唯一值为：{unique_labels}。"
            "本脚本不会执行1/2到0/1映射。"
        )

    X = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce")
    x_array = X.to_numpy(dtype=float)
    if np.isinf(x_array).any():
        bad_cols = X.columns[np.isinf(x_array).any(axis=0)].tolist()
        raise ValueError(f"特征存在 inf，问题列示例：{bad_cols[:10]}")

    n_nan = int(X.isna().sum().sum())
    if n_nan:
        print(f"[数据][提示] 特征中存在 {n_nan} 个 NaN；XGBoost 将按原生缺失值机制处理。")

    print(f"[数据] X={X.shape}，groups={len(np.unique(groups))}，标签0/1={int((y_raw==0).sum())}/{int((y_raw==1).sum())}")
    return {
        "X": X,
        "y": y_raw.reset_index(drop=True),
        "groups": groups,
        "feature_names": feature_names,
        "group_column": group_col,
        "label_column": label_col,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "n_groups": int(len(np.unique(groups))),
    }


def load_existing_cv_results(cv_path: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    print(f"[既有CV] 只读载入：{path_reference(cv_path)}")
    cv_df = pd.read_csv(cv_path)
    required_cols = {"fold", "outer_test_auc", "inner_val_auc", "best_iteration", "best_params"}
    missing = sorted(required_cols.difference(cv_df.columns))
    if missing:
        raise ValueError(f"cv_xgb_results.csv 缺少字段：{missing}")
    if len(cv_df) < 2:
        raise ValueError("cv_xgb_results.csv 有效折数不足。")

    candidates_by_key: Dict[str, Dict[str, Any]] = {}
    for row in cv_df.itertuples(index=False):
        params = parse_params(getattr(row, "best_params"))
        key = canonical_param_key(params)
        fold = int(getattr(row, "fold"))
        if key not in candidates_by_key:
            candidates_by_key[key] = {
                "params": params,
                "source_folds": [fold],
            }
        else:
            candidates_by_key[key]["source_folds"].append(fold)

    candidates = list(candidates_by_key.values())
    for idx, item in enumerate(candidates, start=1):
        item["candidate_id"] = idx

    outer_mean = float(cv_df["outer_test_auc"].mean())
    outer_std = float(cv_df["outer_test_auc"].std(ddof=1))
    print(f"[既有CV] Outer-test AUC = {outer_mean:.6f} ± {outer_std:.6f}")
    print(f"[既有CV] 提取到 {len(candidates)} 组不重复的既有 best_params；不执行新的随机搜索。")
    return cv_df, candidates


# =============================================================================
# 与原训练脚本一致的模型构建和早停兼容逻辑
# =============================================================================
def make_model(
    params: Dict[str, Any],
    n_estimators: int,
    scale_pos_weight: float,
    early_stop: bool,
    seed: int | None = None,
) -> XGBClassifier:
    if seed is None:
        seed = RANDOM_SEED
    kwargs: Dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "min_child_weight": float(params["min_child_weight"]),
        "gamma": float(params["gamma"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "scale_pos_weight": float(scale_pos_weight),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": N_JOBS,
    }

    if early_stop:
        try:
            return XGBClassifier(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **kwargs)
        except TypeError:
            return XGBClassifier(**kwargs)
    return XGBClassifier(**kwargs)


def fit_with_early_stopping(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
    return model


def predict_positive(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    best_it = best_iteration_of(model)
    if best_it is not None and best_it >= 0:
        try:
            return model.predict_proba(X, iteration_range=(0, best_it + 1))[:, 1]
        except TypeError:
            pass
        try:
            return model.predict_proba(X, ntree_limit=best_it + 1)[:, 1]
        except TypeError:
            pass
    return model.predict_proba(X)[:, 1]


# =============================================================================
# 构造固定且有效的全数据内部 group-aware 5折
# =============================================================================
def build_valid_splits(
    y: pd.Series,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    max_retries: int = 50,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str, int]:
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups)
    indices = np.arange(len(y_arr))
    n_groups = len(np.unique(groups_arr))
    if n_groups < n_splits:
        raise ValueError(f"protein_id 分组数 {n_groups} 小于 n_splits={n_splits}。")

    if HAS_SGKF:
        for offset in range(max_retries):
            current_seed = int(seed + offset)
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=current_seed,
            )
            splits = list(splitter.split(indices, y_arr, groups_arr))
            valid = True
            for train_idx, val_idx in splits:
                if len(np.unique(y_arr[train_idx])) < 2 or len(np.unique(y_arr[val_idx])) < 2:
                    valid = False
                    break
                if set(groups_arr[train_idx]).intersection(set(groups_arr[val_idx])):
                    raise RuntimeError("内部划分出现 protein_id 重叠。")
            if valid:
                return splits, "StratifiedGroupKFold", current_seed
        raise ValueError(
            f"尝试 {max_retries} 个随机种子后，仍无法构造所有折均含0/1两类的 StratifiedGroupKFold。"
        )

    # 较老 sklearn 的回退。仍保持 group 不重叠，但不保证分层。
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(indices, y_arr, groups_arr))
    for train_idx, val_idx in splits:
        if len(np.unique(y_arr[train_idx])) < 2 or len(np.unique(y_arr[val_idx])) < 2:
            raise ValueError("GroupKFold 回退划分出现单类验证折；请升级 scikit-learn 以使用 StratifiedGroupKFold。")
        if set(groups_arr[train_idx]).intersection(set(groups_arr[val_idx])):
            raise RuntimeError("内部划分出现 protein_id 重叠。")
    return splits, "GroupKFold", seed


# =============================================================================
# 只评估既有候选参数，不做随机搜索
# =============================================================================
def evaluate_candidates(
    data: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X: pd.DataFrame = data["X"]
    y: pd.Series = data["y"]

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        params = candidate["params"]
        source_folds = candidate["source_folds"]
        print("\n" + "=" * 78)
        print(f"[候选 {candidate_id}/{len(candidates)}] 来源原外层折：{source_folds}")
        print(json.dumps(params, ensure_ascii=False, sort_keys=True))

        fold_aucs: List[float] = []
        fold_tree_counts: List[int] = []

        for inner_fold, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            spw_base = spw_of(y_train)
            scale_pos_weight = float(params["spw_mult"] * spw_base)
            model = make_model(
                params=params,
                n_estimators=N_ESTIMATORS_MAX,
                scale_pos_weight=scale_pos_weight,
                early_stop=True,
                seed=RANDOM_SEED,
            )
            fit_with_early_stopping(model, X_train, y_train, X_val, y_val)
            val_proba = predict_positive(model, X_val)
            val_auc = float(roc_auc_score(y_val, val_proba))
            best_it = best_iteration_of(model)
            tree_count = int(best_it + 1) if best_it is not None and best_it >= 0 else N_ESTIMATORS_MAX

            fold_aucs.append(val_auc)
            fold_tree_counts.append(tree_count)
            detail_rows.append({
                "candidate_id": candidate_id,
                "source_folds": ",".join(map(str, source_folds)),
                "inner_fold": inner_fold,
                "validation_auc": val_auc,
                "best_iteration_zero_based": best_it,
                "n_estimators_actual": tree_count,
                "scale_pos_weight": scale_pos_weight,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            })
            print(
                f"  inner fold {inner_fold}: AUC={val_auc:.6f}, "
                f"best_iteration={best_it}, trees={tree_count}"
            )

        summary_rows.append({
            "candidate_id": candidate_id,
            "source_folds": ",".join(map(str, source_folds)),
            "mean_inner_auc": float(np.mean(fold_aucs)),
            "std_inner_auc": float(np.std(fold_aucs, ddof=1)),
            "median_n_estimators": int(np.median(fold_tree_counts)),
            "min_n_estimators": int(np.min(fold_tree_counts)),
            "max_n_estimators": int(np.max(fold_tree_counts)),
            "n_estimators_per_fold": ",".join(map(str, fold_tree_counts)),
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_inner_auc", "std_inner_auc", "median_n_estimators"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary_df.insert(0, "selection_rank", np.arange(1, len(summary_df) + 1))
    return summary_df, detail_df


# =============================================================================
# 全量训练并保存独立 SHAP 模型
# =============================================================================
def train_and_save_final_model(
    data: Dict[str, Any],
    cv_df: pd.DataFrame,
    cv_path: Path,
    data_path: Path,
    candidate_summary: pd.DataFrame,
    candidate_details: pd.DataFrame,
    split_method: str,
    split_seed: int,
    outdir: Path,
) -> Path:
    if candidate_summary.empty:
        raise RuntimeError("候选参数评估结果为空。")

    winner = candidate_summary.iloc[0]
    selected_params = parse_params(winner["params"])
    final_n_estimators = int(winner["median_n_estimators"])
    if final_n_estimators < 1:
        raise RuntimeError(f"最终树数非法：{final_n_estimators}")

    X: pd.DataFrame = data["X"]
    y: pd.Series = data["y"]
    final_spw = float(selected_params["spw_mult"] * spw_of(y))

    print("\n" + "=" * 78)
    print("[最终模型] 选中的候选参数：")
    print(f"  candidate_id        = {int(winner['candidate_id'])}")
    print(f"  source_folds        = {winner['source_folds']}")
    print(f"  mean_inner_auc      = {float(winner['mean_inner_auc']):.6f}")
    print(f"  std_inner_auc       = {float(winner['std_inner_auc']):.6f}")
    print(f"  trees_per_fold      = {winner['n_estimators_per_fold']}")
    n_inner_folds = len(str(winner["n_estimators_per_fold"]).split(","))
    print(f"  final_n_estimators  = {final_n_estimators}（{n_inner_folds}折中位数）")
    print(json.dumps(selected_params, ensure_ascii=False, sort_keys=True, indent=2))

    final_model = make_model(
        params=selected_params,
        n_estimators=final_n_estimators,
        scale_pos_weight=final_spw,
        early_stop=False,
        seed=RANDOM_SEED,
    )
    final_model.fit(X, y, verbose=False)

    booster = final_model.get_booster()
    used_feature_scores = booster.get_score(importance_type="weight")
    used_features = sorted(used_feature_scores)
    n_used_features = len(used_features)

    # 训练集预测范围只作模型状态描述，不作为性能评估。
    train_proba = final_model.predict_proba(X)[:, 1]
    proba_summary = {
        "min": float(np.min(train_proba)),
        "p01": float(np.percentile(train_proba, 1)),
        "median": float(np.median(train_proba)),
        "p99": float(np.percentile(train_proba, 99)),
        "max": float(np.max(train_proba)),
    }

    outer_mean = float(cv_df["outer_test_auc"].mean())
    outer_std = float(cv_df["outer_test_auc"].std(ddof=1))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "xgb_final_model.joblib"
    summary_path = outdir / "candidate_parameter_evaluation.csv"
    details_path = outdir / "candidate_fold_details.csv"
    meta_path = outdir / "final_model_for_shap_meta.json"

    candidate_summary.to_csv(summary_path, index=False)
    candidate_details.to_csv(details_path, index=False)

    bundle = {
        "model": final_model,
        "features": data["feature_names"],
        "label_map": LABEL_MAP,
        "inverse_label_map": INVERSE_LABEL_MAP,
        "best_params": selected_params,
        "scale_pos_weight": final_spw,
        "final_n_estimators": final_n_estimators,
        "parameter_selection": {
            "rule": (f"Re-evaluate only the unique per-fold best_params already present in cv_xgb_results.csv using full-data internal {INNER_N_SPLITS}-fold {split_method}; select highest mean internal AUC, then lowest standard deviation, then lowest median tree count."),
            "selected_candidate_id": int(winner["candidate_id"]),
            "source_folds": str(winner["source_folds"]),
            "mean_inner_auc": float(winner["mean_inner_auc"]),
            "std_inner_auc": float(winner["std_inner_auc"]),
        },
        "tree_count_estimation": {
            "rule": "Median of best_iteration + 1 across all full-data internal group-aware folds for the selected parameter set.",
            "n_estimators_per_fold": [int(v) for v in str(winner["n_estimators_per_fold"]).split(",")],
            "final_n_estimators": final_n_estimators,
            "split_method": split_method,
            "split_seed": split_seed,
        },
        "original_outer_cv": {
            "cv_results_path": path_reference(cv_path),
            "cv_results_sha256": sha256_file(cv_path),
            "mean_outer_test_auc": outer_mean,
            "std_outer_test_auc": outer_std,
            "note": "These AUC values are read unchanged from the existing outer CV results; the final-model-only script does not recompute or modify them.",
        },
        "trained_on_all_samples": True,
        "deployment_and_shap_only": True,
        "note": "Final single XGBoost refit on all samples for SHAP. Report predictive performance only from the pre-existing outer GroupKFold cv_xgb_results.csv.",
    }
    joblib.dump(bundle, model_path)

    meta = {
        "timestamp": timestamp,
        "data_path": path_reference(data_path),
        "cv_results_path": path_reference(cv_path),
        "cv_results_sha256": sha256_file(cv_path),
        "original_outer_test_auc_mean": outer_mean,
        "original_outer_test_auc_std": outer_std,
        "original_auc_recomputed": False,
        "original_cv_files_modified": False,
        "n_samples": data["n_samples"],
        "n_features": data["n_features"],
        "n_groups": data["n_groups"],
        "group_column": data["group_column"],
        "label_column": data["label_column"],
        "label_values": [0, 1],
        "candidate_count": int(len(candidate_summary)),
        "selected_candidate_id": int(winner["candidate_id"]),
        "selected_source_folds": str(winner["source_folds"]),
        "selected_params": selected_params,
        "selected_mean_inner_auc": float(winner["mean_inner_auc"]),
        "selected_std_inner_auc": float(winner["std_inner_auc"]),
        "n_estimators_per_inner_fold": [int(v) for v in str(winner["n_estimators_per_fold"]).split(",")],
        "final_n_estimators": final_n_estimators,
        "scale_pos_weight": final_spw,
        "split_method": split_method,
        "split_seed": split_seed,
        "n_features_used_by_model": n_used_features,
        "features_used_by_model": used_features,
        "training_probability_summary_not_for_performance_reporting": proba_summary,
        "model_path": path_reference(model_path),
        "candidate_summary_path": path_reference(summary_path),
        "candidate_details_path": path_reference(details_path),
        "interpretation_rule": "Use this model for SHAP only. Continue reporting AUC from the original outer GroupKFold results.",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("[完成] 最终 SHAP 模型已单独保存，不覆盖原模型或原 CV 文件：")
    print(f"  模型：{path_reference(model_path)}")
    print(f"  元数据：{path_reference(meta_path)}")
    print(f"  候选汇总：{path_reference(summary_path)}")
    print(f"  折级明细：{path_reference(details_path)}")
    print(f"  模型实际使用特征数：{n_used_features}/{data['n_features']}")
    print(f"  原 Outer-test AUC 保持：{outer_mean:.6f} ± {outer_std:.6f}")
    if final_n_estimators <= 1:
        print("[警告] 最终树数为1；这是模型状态提示，不改变已选模型。")
    print("[说明] features_used_by_model 来自树分裂使用次数，不是 SHAP 非零特征列表。")
    print("=" * 78)
    return model_path


def main(argv: Sequence[str] | None = None) -> None:
    global RANDOM_SEED, INNER_N_SPLITS, N_ESTIMATORS_MAX, EARLY_STOPPING_ROUNDS, N_JOBS
    args, settings = load_cli_config(argv)
    RANDOM_SEED = settings["random_seed"]
    INNER_N_SPLITS = settings["n_splits"]
    N_ESTIMATORS_MAX = settings["n_estimators_max"]
    EARLY_STOPPING_ROUNDS = settings["early_stopping_rounds"]
    N_JOBS = settings["n_jobs"]
    np.random.seed(RANDOM_SEED)

    prepare_output_directory(args.outdir)
    metadata = write_run_context(args, settings)
    write_json(args.outdir / "run_metadata.json", metadata)
    try:
        data_path, cv_path, outdir = args.data_path, args.cv_path, args.outdir
        print("=" * 78)
        print(f"{TASK}: final-model-only XGBoost refit for later SHAP")
        print("Does not compute SHAP values or rerun outer-test evaluation.")
        print(f"Data: {path_reference(data_path)}")
        print(f"Existing CV: {path_reference(cv_path)}")
        print(f"Output: {path_reference(outdir)}")
        print("=" * 78)

        cv_hash_before = metadata["cv_results_sha256_before"]
        data = load_data(data_path)
        cv_df, candidates = load_existing_cv_results(cv_path)
        splits, split_method, split_seed = build_valid_splits(
            y=data["y"], groups=data["groups"], n_splits=INNER_N_SPLITS,
            seed=RANDOM_SEED, max_retries=settings["max_split_retries"],
        )
        print(f"[内部划分] 方法={split_method}，折数={len(splits)}，random_state={split_seed}")
        candidate_summary, candidate_details = evaluate_candidates(data, candidates, splits)
        model_path = train_and_save_final_model(
            data=data, cv_df=cv_df, cv_path=cv_path, data_path=data_path,
            candidate_summary=candidate_summary, candidate_details=candidate_details,
            split_method=split_method, split_seed=split_seed, outdir=outdir,
        )
        cv_hash_after = sha256_file(cv_path)
        metadata["cv_results_sha256_after"] = cv_hash_after
        if cv_hash_before != cv_hash_after:
            raise RuntimeError("cv_xgb_results.csv changed during the run; check input provenance.")
        metadata.update({
            "status": "completed", "original_cv_unchanged": True,
            "model": path_reference(model_path), "model_sha256": sha256_file(model_path),
            "n_samples": data["n_samples"], "n_features": data["n_features"],
            "n_groups": data["n_groups"], "n_candidates": len(candidates),
            "internal_split_method": split_method, "internal_split_seed": split_seed,
        })
        print("[完整性] cv_xgb_results.csv SHA256 前后一致，原AUC结果文件未被修改。")
        print("[范围] 已完成解释模型重训；尚未计算 SHAP 值或生成 SHAP 图。")
    except Exception as exc:
        metadata["status"] = "failed"
        # Do not serialize the exception message; it can contain private paths/data.
        metadata["exception_type"] = type(exc).__name__
        raise
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(args.outdir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()
