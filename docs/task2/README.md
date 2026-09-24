# Task 2: grouped binary-classifier training

This add-on prepares the seven supplied Task 2 training scripts for inclusion
in the same repository as Task 1. **It adds Task 2 files without replacing
Task 1 scripts, configs, root README files, or root requirements.** It does
not include the research CSV, fitted models, raw source copies, or test logs.

[中文说明](README_zh-CN.md)

## Placement and installation

Copy the **contents** of `task2_github_release/` into the repository root,
merging the directory trees. Do not put the wrapper folder inside
`scripts/training/` or `configs/`.

```text
repository/
├── scripts/training/
│   ├── task1/                 # existing; leave unchanged
│   └── task2/                 # seven train_*.py files and _runtime.py
├── configs/
│   ├── task1/                 # existing; leave unchanged
│   └── task2/                 # seven model JSON files
├── data/task2/
│   ├── .gitignore
│   └── README.md
└── docs/task2/                # this guide, changes, dependencies, verification
```

Keep `_runtime.py` inside `scripts/training/task2/`. The default config paths
are located from the script's repository layout, so `scripts/` and `configs/`
must remain together. No Task 1 helper file is overwritten or required.

The previous Task 1 package lists the same direct dependencies. For a separate
local environment, run from the repository root:

```bash
python -m pip install -r docs/task2/requirements-tested.txt
```

These pins describe the environment exercised here, not the original paper
environment. They are not a complete transitive lockfile. The unpinned direct
requirements are also available in `docs/task2/requirements.txt`; other package
versions have not been validated by this release. SHAP is optional and is not
part of the tested Task 2 execution path.

## Run one classifier

Default input:

```text
data/task2/proid_1_2_feature_labels_standard.csv
```

Run from the repository root:

```bash
python scripts/training/task2/train_xgboost.py
```

Explicit paths and supported estimator parallelism:

```bash
python scripts/training/task2/train_xgboost.py --config configs/task2/xgboost.json --data data/task2/proid_1_2_feature_labels_standard.csv --output-dir results/task2/xgboost/run_01 --n-jobs -1
```

An input CSV stored in the current working directory may be supplied directly:

```bash
python scripts/training/task2/train_xgboost.py --data proid_1_2_feature_labels_standard.csv
```

Each script automatically loads its matching JSON; no `--config` is needed
for the published defaults.

| Classifier | Script in `scripts/training/task2/` | JSON in `configs/task2/` |
|---|---|---|
| XGBoost | `train_xgboost.py` | `xgboost.json` |
| Logistic regression | `train_logistic_regression.py` | `logistic_regression.json` |
| Gaussian naive Bayes | `train_gaussian_nb.py` | `gaussian_nb.json` |
| K-nearest neighbors | `train_knn.py` | `knn.json` |
| Linear SVM (`LinearSVC`) | `train_linear_svm.py` | `linear_svm.json` |
| Decision tree | `train_decision_tree.py` | `decision_tree.json` |
| Random forest | `train_random_forest.py` | `random_forest.json` |

All seven command-line interfaces support `--config`, `--data`, `--output-dir`,
`--n-jobs`, `--n-trials`, `--seed`, and `--dry-run`. Changing trials or the seed
changes the published experiment settings. `--n-jobs -1` preserves the source
estimator parallelism; it does not create distributed execution or parallel
outer folds. Positive values also cap numerical thread pools. Models without
an estimator-level `n_jobs` interface do not acquire one.

`--dry-run` prints the effective settings and whether the input file exists,
without reading the CSV, training, or creating results. It is not a validation
of the dataset's contents. Entry points are intended to be launched separately
as shown above, not to run Task 1 and Task 2 scripts in one shared interpreter.

## Paths and outputs

JSON paths are relative to the repository root. Explicit relative CLI paths
are relative to the shell's current working directory. Local absolute input
paths are accepted through `--data` without embedding them into public code.

Without `--output-dir`, each run creates:

```text
results/task2/<model>/<YYYYMMDD_HHMMSS_microseconds>/
```

An explicitly supplied results directory must be absent or empty. The wrapper
refuses to overwrite a populated directory. It does not modify Task 1 results.
Source result filenames and CSV column names are preserved. In addition,
`run_config.json`, `environment.json`, and `run_metadata.json` record effective
settings, observed versions, input/code SHA-256 digests, and run status.

External absolute directories are redacted in shareable JSON metadata as
`<external>/<filename>`. This intentionally loses the private directory name;
such redacted files are records, not automatically portable input configs.
Original console messages and exception tracebacks may still contain local
output paths; no raw logs are included in the release.

## Labels, input columns, and score direction

The input schema remains positional: first column is the group, middle columns
are numeric features, final column is the original `1/2` label. No automatic
metadata-column selection, restandardization, or imputation is added.

```python
LABEL_MAP = {1: 0, 2: 1}
INVERSE_LABEL_MAP = {0: 1, 1: 2}
```

Class `2` is the original positive class. Exported `true_label` and `pred_label`
remain `1/2`. XGBoost, NB, KNN, and DT retain `pred_proba`; RF and LR retain
`pred_proba_class_2`. The SVM retains `decision_score`, not a calibrated
probability; larger scores favor original class `2`.

## Preserved source protocol

The defaults retain five outer GroupKFold folds and one group-aware inner
train/validation holdout per outer fold. The inner holdout uses the first
StratifiedGroupKFold split, with the original fallback and retry rules. It is
not an inner five-fold average. The default candidate budget is 50 per outer
fold, with each classifier's own source candidate-generation order.

Hyperparameters are selected using inner validation AUC. The CV estimator is
fitted on inner-training samples; validation samples are not added to a new
outer-training refit. XGBoost uses that validation split for early stopping.
Original near-constant feature filtering, final-model fitting, and per-fold
metric definitions are retained. The separately saved full-data model does
not provide the cross-validation performance reported in the source workflow.

XGBoost's `BASELINE_AUC = 0.7931` is a historical reference embedded in the
supplied source. It is preserved in `configs/task2/xgboost.json` only for the
existing summary comparison; it is not computed in this release and does not
participate in model fitting or hyperparameter selection. Its provenance has
not been independently verified. Setting it to JSON `null` disables this
optional historical comparison without changing the fitted model.

The source's blanket warning suppression is not retained; convergence and
compatibility warnings remain visible. Existing LR SHAP logic is retained but
imported only when its `RUN_SHAP` switch is true (default false).

## Verification and limitations

All seven scripts passed source/new comparisons on 240 synthetic samples from
20 groups, with mixed labels within each group and one constant feature. All
five outer folds and final-model fitting were executed using three candidates
per fold. XGBoost used a test-only 60-tree cap and 8-round early stopping; both
sides used the same numerical thread cap. Default release configs remain at
50 candidates, 3000 trees and 100 early-stopping rounds for XGBoost.

All generated candidate sequences under the five original fold seeds were
separately compared at the full 50-candidate budget. Numeric CSV/JSON outputs
and fitted-model predictions matched within absolute and relative tolerances
of `1e-12`. Images were produced by the source routines but were not compared
pixel-by-pixel. See [VERIFICATION.md](VERIFICATION.md).

No real Task 2 CSV was supplied, and the full paper-scale runs, optional SHAP,
legacy dependency fallbacks, upstream standardization, scientific labeling
rules, and whole-study leakage status have not been verified here. This is a
publication refactor and bounded regression check, not a revalidation of the
paper's conclusions.
