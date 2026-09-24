"""Small publication interface shared by the seven Task 2 training scripts.

This module handles JSON configuration, paths, thread limits, and run metadata.
It does not implement data splitting, model selection, scoring, or relabeling.
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
from typing import Any, Callable, Sequence


PACKAGE_NAMES = (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib",
    "matplotlib", "seaborn", "threadpoolctl", "shap",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object; configuration files are never executed."""
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("The configuration must be a JSON object.")
    return value


def load_default_config(script_file: str, model_name: str):
    """Locate configs relative to the repository, independently of the shell cwd."""
    root = Path(script_file).resolve().parents[3]
    path = root / "configs" / "task2" / f"{model_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing configs/task2/{model_name}.json. Keep scripts/ and configs/ "
            "together in the supplied repository layout."
        )
    config = read_json(path)
    if config.get("model") != model_name or config.get("task") != "task2":
        raise ValueError("The default configuration has the wrong task or model.")
    return config, root


def resolve_path(value: str, base: Path) -> Path:
    """Config paths use the repository root; explicit CLI paths use the shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def public_path(value: str | Path, root: Path) -> str:
    """Do not write external absolute directory names into shareable JSON output."""
    path = Path(value).resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_config(config: dict, reference: dict, model_name: str) -> None:
    """Reject misspelled or incompatible settings rather than silently ignoring them."""
    if set(config) != set(reference):
        raise ValueError(
            "Configuration sections must match the supplied model JSON. "
            f"Missing: {sorted(set(reference) - set(config))}; "
            f"unknown: {sorted(set(config) - set(reference))}."
        )
    if config["task"] != "task2" or config["model"] != model_name:
        raise ValueError(f"Expected task2 configuration for {model_name}.")
    for section in ("data", "output", "runtime", "parameters", "search_space", "baseline_params", "fixed_model_params"):
        if not isinstance(config[section], dict) or set(config[section]) != set(reference[section]):
            raise ValueError(f"Keys in '{section}' must match the supplied {model_name}.json.")
    for section, key in (("data", "path"), ("output", "root")):
        if not isinstance(config[section][key], str) or not config[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty path string.")

    for name, original in reference["parameters"].items():
        value = config["parameters"][name]
        if name == "BASELINE_AUC":
            valid = value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)
        elif isinstance(original, bool):
            valid = isinstance(value, bool)
        elif isinstance(original, int):
            valid = type(value) is int
        elif isinstance(original, float):
            valid = type(value) in (int, float) and math.isfinite(value)
        elif original is None:
            valid = value is None or (type(value) in (int, float) and math.isfinite(value))
        else:
            valid = isinstance(value, type(original))
        if not valid:
            raise ValueError(f"Invalid type/value for {name}.")
    parameters = config["parameters"]
    for name in ("N_FOLDS", "INNER_N_SPLITS"):
        if parameters[name] < 2:
            raise ValueError(f"{name} must be at least 2.")
    for name in ("N_TRIALS", "INNER_MAX_RETRIES", "N_ESTIMATORS", "EARLY_STOPPING_ROUNDS", "SHAP_SAMPLE_N"):
        if name in parameters and parameters[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if not 0 < parameters["INNER_VALID_SIZE"] < 1:
        raise ValueError("INNER_VALID_SIZE must lie strictly between 0 and 1.")
    if parameters["NEAR_CONST_STD_TOL"] < 0 or parameters["RANDOM_SEED"] < 0:
        raise ValueError("NEAR_CONST_STD_TOL and RANDOM_SEED must be nonnegative.")
    jobs = config["runtime"]["n_jobs"]
    if type(jobs) is not int or (jobs != -1 and jobs < 1):
        raise ValueError("n_jobs must be -1 or a positive integer.")
    for name, values in config["search_space"].items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_space.{name} must be a nonempty list.")


def apply_config(namespace: dict, config: dict) -> None:
    """Expose validated settings to the unchanged, model-specific training functions."""
    namespace.update(config["parameters"])
    namespace["N_JOBS"] = config["runtime"]["n_jobs"]
    namespace["BASELINE_PARAMS"] = deepcopy(config["baseline_params"])
    namespace["FIXED_MODEL_PARAMS"] = deepcopy(config["fixed_model_params"])
    if config["model"] == "knn":
        for variable, name in (("K_VALUES", "n_neighbors"), ("WEIGHT_VALUES", "weights"), ("METRIC_VALUES", "metric")):
            namespace[variable] = list(config["search_space"][name])
    else:
        # Preserve JSON key order: the original samplers use dictionary order.
        namespace["SEARCH_SPACE"] = deepcopy(config["search_space"])


def run_cli(
    *,
    script_file: str,
    model_name: str,
    namespace: dict,
    training_function: Callable[[], Any],
    argv: Sequence[str] | None = None,
):
    """Execute one model; a fresh output directory is required for every run."""
    root = namespace["REPO_ROOT"]
    reference = namespace["DEFAULT_CONFIG"]
    parser = argparse.ArgumentParser(
        description=f"Task 2 {model_name}: grouped outer CV and inner holdout tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help=f"Complete JSON configuration; default: configs/task2/{model_name}.json")
    parser.add_argument("--data", help="Input CSV. An explicit relative path is relative to the shell working directory.")
    parser.add_argument("--output-dir", help="Exact output directory, not a parent directory; must be new or empty.")
    parser.add_argument("--n-jobs", type=int, help="-1 keeps original parallelism; positive values cap numerical threads and supported estimator jobs.")
    parser.add_argument("--n-trials", type=int, help="Override N_TRIALS; changing this is not the default paper protocol.")
    parser.add_argument("--seed", type=int, help="Override RANDOM_SEED; changing this changes splits/sampling.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/print settings without loading data, creating outputs, or training.")
    args = parser.parse_args(argv)

    config_file = (
        resolve_path(args.config, Path.cwd()) if args.config
        else root / "configs" / "task2" / f"{model_name}.json"
    )
    try:
        config = read_json(config_file)
        validate_config(config, reference, model_name)
        if args.n_jobs is not None:
            config["runtime"]["n_jobs"] = args.n_jobs
        if args.n_trials is not None:
            config["parameters"]["N_TRIALS"] = args.n_trials
        if args.seed is not None:
            config["parameters"]["RANDOM_SEED"] = args.seed
        validate_config(config, reference, model_name)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    data_path = resolve_path(args.data, Path.cwd()) if args.data else resolve_path(config["data"]["path"], root)
    if args.output_dir:
        output_path = resolve_path(args.output_dir, Path.cwd())
    else:
        output_path = resolve_path(config["output"]["root"], root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    effective = deepcopy(config)
    effective["data"]["path"] = public_path(data_path, root)
    effective["output"]["root"] = public_path(output_path.parent, root)

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "input_file_exists": data_path.is_file(),
            "run_directory": public_path(output_path, root),
            "configuration": effective,
            "note": "No data loaded and no outputs created. <external> means the local directory was redacted.",
        }, ensure_ascii=False, indent=2, allow_nan=False))
        return None
    if not data_path.is_file():
        parser.error(f"Input CSV not found: {public_path(data_path, root)}. Provide --data or place the file at the configured location.")

    # No recursive deletion, automatic overwriting, or reuse of populated results.
    if output_path.exists():
        if not output_path.is_dir() or next(output_path.iterdir(), None) is not None:
            parser.error("Output directory already exists and is not empty. Use a new --output-dir or omit it for a timestamped run.")
    else:
        output_path.mkdir(parents=True, exist_ok=False)

    apply_config(namespace, config)
    namespace["DATA_PATH"] = str(data_path)
    namespace["DATA_REFERENCE"] = public_path(data_path, root)
    namespace["OUTDIR"] = str(output_path)
    random.seed(namespace["RANDOM_SEED"])
    namespace["np"].random.seed(namespace["RANDOM_SEED"])
    # Do not globally suppress convergence/deprecation warnings in public scripts.

    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output_path / "run_config.json", effective)
    write_json(output_path / "environment.json", {
        "python": platform.python_version(),
        "packages": versions,
        "note": "Observed runtime versions, not a claim about the historical paper environment.",
    })
    provenance = {
        "model": model_name,
        "task": "task2",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_reference": public_path(data_path, root),
        "output_reference": public_path(output_path, root),
        "input_sha256": sha256_file(data_path),
        "input_size_bytes": data_path.stat().st_size,
        "entrypoint": public_path(script_file, root),
        "entrypoint_sha256": sha256_file(Path(script_file)),
        "runtime_helper_sha256": sha256_file(Path(__file__)),
        "configuration_reference": public_path(config_file, root),
        "source_configuration_sha256": sha256_file(config_file),
        "local_absolute_paths_recorded": False,
        "status": "running",
    }
    write_json(output_path / "run_metadata.json", provenance)

    jobs = config["runtime"]["n_jobs"]
    if jobs > 0:
        from threadpoolctl import threadpool_limits
        numerical_threads = threadpool_limits(limits=jobs)
    else:
        numerical_threads = nullcontext()
    try:
        with numerical_threads:
            result = training_function()
        provenance["status"] = "completed"
        return result
    except BaseException as error:
        provenance["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        provenance["exception_type"] = type(error).__name__
        raise
    finally:
        provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_path / "run_metadata.json", provenance)
