"""Path/configuration interface for the Task 2 evaluation release.

No model fitting, resampling, scoring or label mapping is implemented here.
Config-relative paths use the repository root; explicit CLI paths use the cwd.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib",
            "matplotlib", "seaborn", "threadpoolctl")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object.")
    return value


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def public_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def default_config(category: str, name: str) -> tuple[dict, Path]:
    path = REPO_ROOT / "configs" / "task2" / "evaluation" / category / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing configs/task2/evaluation/{category}/{name}.json. "
            "Keep the supplied scripts/ and configs/ layout together."
        )
    return read_json(path), path


def _check_keys(actual: Any, reference: Any, prefix: str = "config") -> None:
    # Lists retain their original order; this matters for candidate generation.
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError(f"Keys in {prefix} must match the supplied default JSON.")
        for key in reference:
            _check_keys(actual[key], reference[key], f"{prefix}.{key}")


def _settings(parser, args, category, name):
    reference, path = default_config(category, name)
    if args.config:
        path = resolve_path(args.config, Path.cwd())
    config = read_json(path)
    _check_keys(config, reference)
    for key in ("task", "analysis"):
        if config[key] != reference[key]:
            raise ValueError(f"{key} must be {reference[key]!r}.")
    identity_key = "model" if category == "bootstrap" else "variant"
    if config[identity_key] != name:
        raise ValueError(f"This entrypoint requires {identity_key}={name!r}.")
    return config, path


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def check_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or next(path.iterdir(), None) is not None):
        raise FileExistsError("Output directory is not empty. Choose a new --output-dir; existing results are never overwritten.")


def _check_output_file(path: Path, input_path: Path) -> None:
    if path == input_path:
        raise ValueError("Input and output paths must be different.")
    if path.exists() or path.with_suffix(path.suffix + ".metadata.json").exists():
        raise FileExistsError("Output CSV or its metadata already exists. Choose a new --output-file.")


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def environment() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions,
            "note": "Observed runtime environment, not the historical paper environment."}


def _file_record(path: Path) -> dict:
    # Record references/size, not data contents or environment variables.
    return {"path": public_path(path), "size_bytes": path.stat().st_size}


def save_run_info(output_dir: Path, args, script_file: str, inputs) -> None:
    """Write successful-run records; no console log or exception text is captured."""
    output_dir = Path(output_dir)
    effective = deepcopy(args.effective_config)
    write_json(output_dir / "run_config.json", effective)
    write_json(output_dir / "environment.json", environment())
    write_json(output_dir / "run_metadata.json", {
        "task": "task2", "analysis": effective["analysis"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "entrypoint": public_path(script_file),
        "entrypoint_sha256": hashlib.sha256(Path(script_file).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": public_path(args.config_path),
        "output": public_path(output_dir),
        "inputs": [_file_record(Path(p)) for p in inputs if Path(p).is_file()],
        "privacy_note": "External directory prefixes are redacted; basenames and scientific feature names are retained. Console logs are not sanitized by this helper.",
    })


def bootstrap_args(script_file: str, model: str, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Task 2 {model}: protein-cluster bootstrap of saved outer-test predictions.")
    parser.add_argument("--config", help=f"Default: configs/task2/evaluation/bootstrap/{model}.json")
    parser.add_argument("--result-dir", type=Path, help="Directory containing one completed CV run (not the parent of multiple runs).")
    parser.add_argument("--output-dir", type=Path, help="New/empty output directory; default: <result-dir>/bootstrap.")
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--max-redraw", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print effective settings without reading predictions or writing results.")
    args = parser.parse_args(argv)
    try:
        config, path = _settings(parser, args, "bootstrap", model)
        p = config["parameters"]
        for key in ("n_bootstrap", "seed", "confidence", "max_redraw"):
            override = getattr(args, key)
            if override is not None:
                p[key] = override
            setattr(args, key, p[key])
        _integer(p["n_bootstrap"], "n_bootstrap", 100)
        _integer(p["seed"], "seed")
        _integer(p["max_redraw"], "max_redraw")
        _number(p["confidence"], "confidence")
        if not 0 < p["confidence"] < 1:
            raise ValueError("confidence must lie in (0, 1).")
        if "consistency_tol" in p:
            _number(p["consistency_tol"], "consistency_tol")
            if p["consistency_tol"] < 0:
                raise ValueError("consistency_tol must be nonnegative.")
            args.consistency_tol = p["consistency_tol"]
        if args.result_dir is None:
            if not config["input"]["result_dir"]:
                raise ValueError("Provide --result-dir for the completed run, or set input.result_dir in the JSON. No latest-run auto-selection is performed.")
            result_dir = resolve_path(config["input"]["result_dir"], REPO_ROOT)
        else:
            result_dir = resolve_path(args.result_dir, Path.cwd())
        if args.output_dir is not None:
            output_dir = resolve_path(args.output_dir, Path.cwd())
        elif config["output"]["directory"]:
            output_dir = resolve_path(config["output"]["directory"], REPO_ROOT)
        else:
            output_dir = result_dir / "bootstrap"
        args.result_dir, args.output_dir, args.config_path = result_dir, output_dir, path
        effective = deepcopy(config)
        effective["input"]["result_dir"] = public_path(result_dir)
        effective["output"]["directory"] = public_path(output_dir)
        args.effective_config = effective
        if not args.dry_run:
            if not result_dir.is_dir():
                raise FileNotFoundError("Input run directory does not exist: " + public_path(result_dir))
            check_empty_directory(output_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return args


def apply_ablation_config(namespace: dict, config: dict) -> None:
    namespace.update(config["parameters"])
    namespace["N_JOBS"] = config["runtime"]["n_jobs"]
    for key, section in (("SEARCH_SPACE", "search_space"), ("BASELINE_PARAMS", "baseline_params"),
                         ("FIXED_MODEL_PARAMS", "fixed_model_params")):
        namespace[key] = deepcopy(config[section])


def ablation_args(variant: str, stage: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Task 2 XGBoost {variant}: {stage}.")
    parser.add_argument("--config", help=f"Default: configs/task2/evaluation/ablation/{variant}.json")
    parser.add_argument("--data", type=Path, help="Input CSV: full table for prepare; already-ablated table for train.")
    if stage == "prepare":
        parser.add_argument("--output-file", type=Path, help="New output CSV; source data are never overwritten.")
    else:
        parser.add_argument("--output-dir", type=Path, help="New/empty output directory. Default creates a timestamped run.")
        parser.add_argument("--n-jobs", type=int, help="-1 retains original parallelism; a positive integer limits threads.")
        parser.add_argument("--n-trials", type=int)
        parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config, path = _settings(parser, args, "ablation", variant)
        if stage == "train":
            for attr, section, key in (("n_jobs", "runtime", "n_jobs"), ("n_trials", "parameters", "N_TRIALS"), ("seed", "parameters", "RANDOM_SEED")):
                if getattr(args, attr) is not None:
                    config[section][key] = getattr(args, attr)
        p = config["parameters"]
        for key in ("N_TRIALS", "N_ESTIMATORS", "EARLY_STOPPING_ROUNDS", "INNER_MAX_RETRIES"):
            _integer(p[key], key, 1)
        for key in ("N_FOLDS", "INNER_N_SPLITS"):
            _integer(p[key], key, 2)
        _integer(p["RANDOM_SEED"], "RANDOM_SEED")
        for key in ("USE_SCALE_POS_WEIGHT", "DROP_NEAR_CONSTANT"):
            if type(p[key]) is not bool:
                raise ValueError(f"{key} must be a JSON boolean.")
        for key in ("INNER_VALID_SIZE", "NEAR_CONST_STD_TOL"):
            _number(p[key], key)
        if not 0 < p["INNER_VALID_SIZE"] < 1 or p["NEAR_CONST_STD_TOL"] < 0:
            raise ValueError("Invalid inner validation fraction or near-constant tolerance.")
        if p["BASELINE_AUC"] is not None:
            _number(p["BASELINE_AUC"], "BASELINE_AUC")
        jobs = config["runtime"]["n_jobs"]
        if type(jobs) is not int or (jobs != -1 and jobs < 1):
            raise ValueError("n_jobs must be -1 or a positive integer.")
        for name, values in config["search_space"].items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"search_space.{name} must be a nonempty list.")
        spec = config["preparation"]
        if spec["mode"] not in ("explicit", "regex"):
            raise ValueError("preparation.mode must be explicit or regex.")
        _integer(spec["expected_count"], "expected_count", 1)
        if spec["mode"] == "explicit" and (not isinstance(spec["columns"], list) or not spec["columns"]):
            raise ValueError("Explicit removal requires a nonempty columns list.")
        if spec["mode"] == "regex" and not isinstance(spec["pattern"], str):
            raise ValueError("Regex removal requires a pattern string.")
        input_key = "full_table" if stage == "prepare" else "ablated_table"
        data_path = resolve_path(args.data, Path.cwd()) if args.data else resolve_path(config["data"][input_key], REPO_ROOT)
        if stage == "prepare":
            output_path = resolve_path(args.output_file, Path.cwd()) if args.output_file else resolve_path(config["data"]["ablated_table"], REPO_ROOT)
        else:
            output_path = resolve_path(args.output_dir, Path.cwd()) if args.output_dir else resolve_path(config["output"]["root"], REPO_ROOT) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.data_path, args.output_path, args.config_path = data_path, output_path, path
        args.config_values = config
        effective = deepcopy(config)
        for key in ("full_table", "ablated_table"):
            effective["data"][key] = public_path(resolve_path(config["data"][key], REPO_ROOT))
        effective["data"][input_key] = public_path(data_path)
        effective["output"]["root"] = public_path(resolve_path(config["output"]["root"], REPO_ROOT))
        effective["stage"] = stage
        effective["resolved_output"] = public_path(output_path)
        if stage == "prepare":
            effective["data"]["ablated_table"] = public_path(output_path)
        args.effective_config = effective
        if not args.dry_run:
            if not data_path.is_file():
                raise FileNotFoundError("Input CSV missing: " + public_path(data_path))
            if stage == "prepare":
                _check_output_file(output_path, data_path)
            else:
                check_empty_directory(output_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return args


def run_ablation_cli(script_file, variant, namespace, training_function, argv=None):
    args = ablation_args(variant, "train", argv)
    if args.dry_run:
        print(json.dumps(args.effective_config, ensure_ascii=False, indent=2))
        return None
    config = args.config_values
    apply_ablation_config(namespace, config)
    namespace["DATA_PATH"] = str(args.data_path)
    namespace["DATA_REFERENCE"] = public_path(args.data_path)
    namespace["OUTDIR"] = str(args.output_path)
    random.seed(namespace["RANDOM_SEED"])
    namespace["np"].random.seed(namespace["RANDOM_SEED"])
    args.output_path.mkdir(parents=True, exist_ok=True)
    if namespace["N_JOBS"] > 0:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=namespace["N_JOBS"])
    else:
        context = nullcontext()
    with context:
        result = training_function()
    save_run_info(args.output_path, args, script_file, [args.data_path])
    return result


def save_preparation_info(args, script_file, removed, n_rows, n_columns):
    write_json(args.output_path.with_suffix(args.output_path.suffix + ".metadata.json"), {
        "task": "task2", "stage": "prepare_ablation", "status": "completed",
        "configuration": args.effective_config,
        "entrypoint": public_path(script_file),
        "input": _file_record(args.data_path), "output": public_path(args.output_path),
        "removed_columns": list(removed), "n_removed": len(removed),
        "n_rows": int(n_rows), "n_columns_after_removal": int(n_columns),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "note": "Only selected columns were removed. Remaining column order, rows and supplied labels were not intentionally transformed.",
    })
