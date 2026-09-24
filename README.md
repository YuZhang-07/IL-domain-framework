# Task 1 — grouped classifier training

Publication-oriented copies of seven supplied research training scripts. This
package separates executable code from experiment settings without replacing the
model-specific training procedures.

**Scope:** training from an existing feature/label CSV. The research dataset,
upstream standardization/label-generation code, trained models, and paper results
are not included in this package. No biological meaning is inferred for label 1.

中文说明见 [README_zh-CN.md](README_zh-CN.md)。

## Layout

```text
scripts/training/task1/
    _runtime.py
    train_xgboost.py
    train_logistic_regression.py
    train_gaussian_nb.py
    train_knn.py
    train_linear_svm.py
    train_decision_tree.py
    train_random_forest.py
configs/task1/
    xgboost.json
    logistic_regression.json
    gaussian_nb.json
    knn.json
    linear_svm.json
    decision_tree.json
    random_forest.json
data/task1/
    README.md
```

Keep `_runtime.py` alongside all seven training scripts. Keep `scripts/` and
`configs/` under the same repository root. These files are not seven standalone
scripts that can be moved independently of the configuration/helper files.

## Environment

`requirements.txt` lists direct dependencies without claiming a historical lock.
`requirements-tested.txt` pins the packages used for the local reduced-workload
comparison on Python 3.13.5. This tested environment is **not evidence of the
software versions used to produce the original manuscript results**.

For a new environment matching the recorded comparison:

```bash
python -m pip install -r requirements-tested.txt
```

The existing research environment can also be used. Runtime package versions are
recorded automatically. SHAP is optional for logistic regression, remains disabled
by default, and was not exercised in the reduced-workload comparison.

## Input contract

The CSV layout is inherited from the original scripts:

```text
protein_id,feature_1,feature_2,...,feature_N,label
```

The first column supplies grouping IDs, the last column supplies the existing 0/1
labels, and **every intermediate column is used as a feature**. Do not insert
`frame_id`, a saved dataframe index, or other metadata between the first and last
columns unless it is intentionally a predictor. Preserve the original row order
when reproducing an experiment.

Both labels may occur within the same protein. These scripts do not collapse
labels to a single label per protein, merge a third class into label 1, relabel
samples, or add a scaler. They use the input feature values as supplied. Upstream
standardization parameters and the physical meaning of the labels cannot be
reconstructed from the seven training scripts alone.

The original missing-value behavior is retained: XGBoost permits coerced NaN
values; the six scikit-learn scripts reject NaN. All reject infinite features.
This package does not add imputation or claim that upstream preprocessing is
leakage-free.

## Run one model

From the repository root, place the existing input CSV at:

```text
data/task1/proid_0_1_feature_labels_standard.csv
```

Then run:

```bash
python scripts/training/task1/train_xgboost.py
```

The script automatically loads `configs/task1/xgboost.json`. No editing of Python
source is necessary to change the input/output location:

```bash
python scripts/training/task1/train_xgboost.py \
  --config configs/task1/xgboost.json \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --output-dir results/task1/xgboost/run_01 \
  --n-jobs -1
```

Relative paths **inside JSON** are resolved from the repository root. Explicit
relative `--config`, `--data`, and `--output-dir` arguments are resolved from the
shell's current working directory. Local absolute paths are accepted as CLI
arguments but are not embedded in the distributed source/configuration defaults.

Without `--output-dir`, each run creates:

```text
results/task1/<model>/<YYYYMMDD_HHMMSS_microseconds>/
```

An explicit output directory must be nonexistent or empty. A populated directory
is rejected; the code does not automatically delete or overwrite an earlier run.
The timestamps use the executing machine's local time; provenance timestamps
are recorded in UTC.

Other model entry points (run the model(s) needed, not all seven concurrently with
unlimited threads on the same machine):

```bash
python scripts/training/task1/train_logistic_regression.py
python scripts/training/task1/train_gaussian_nb.py
python scripts/training/task1/train_knn.py
python scripts/training/task1/train_linear_svm.py
python scripts/training/task1/train_decision_tree.py
python scripts/training/task1/train_random_forest.py
```

## Configuration and CLI

Each JSON contains `data`, `output`, `runtime`, `parameters`, `search_space`,
`baseline_params`, and `fixed_model_params`. The settings are actually read by the
scripts; they are not documentation-only copies. A custom `--config` is a complete
configuration for the selected model, not a partial override file.

`--data`, `--output-dir`, `--n-jobs`, `--n-trials`, and `--seed` override the
corresponding run settings. Changing the seed, number of trials, search space, or
cross-validation settings changes the experiment. The shipped defaults preserve
42 / 5 outer folds / 50 requested candidates per fold.

`--n-jobs -1` preserves the original estimator parallelism. For positive values,
XGBoost/RF/KNN also receive that estimator job count, and numerical-library
threads are capped for all models. LinearSVC, GaussianNB, LR, and DT do not gain
an estimator-level parallel implementation through this flag. A resource limit
is not a guarantee that every allocated core will be used.

```bash
python scripts/training/task1/train_xgboost.py --help
python scripts/training/task1/train_xgboost.py --dry-run
```

`--dry-run` validates and prints settings without loading the dataset, creating an
output directory, or training. `input_file_exists: false` is expected when the
research CSV has not been supplied.

## Retained scientific procedure

Outer evaluation uses `GroupKFold`. Each outer-training fold produces **one**
group-aware inner training/validation holdout, with the original
`StratifiedGroupKFold` / `GroupShuffleSplit` fallback and retry logic. This is not
an average over five independent inner-validation folds. Near-constant feature
selection remains based on the outer-training fold. Selection uses inner
validation AUC. The evaluated fold model is fitted on inner-training samples,
not refitted on the entire outer-training fold.

Original candidate order, baseline insertion, random generators, search spaces,
XGBoost early stopping/class weighting, and final-model selection are retained.
The scripts implement their own candidate sampling/enumeration, not Optuna.
Default candidate order was compared to the originals for every outer-fold seed.

Final models retain the original model-specific behavior: RF/LR remove
near-constant features on the full dataset; the other scripts keep their original
full-data feature handling. XGBoost estimates its final tree count on a separate
group-aware holdout before fitting the final full-data model. Final-model fits do
not replace held-out CV scores.

## Outputs

Original result filenames and prediction-column conventions are retained:

| Model | CV table | AUC score column in fold predictions |
|---|---|---|
| XGBoost | `cv_xgb_results.csv` | `pred_proba` |
| Logistic regression | `cv_logistic_regression_results.csv` | `pred_proba_class_1` |
| Gaussian NB | `cv_gaussian_nb_results.csv` | `pred_proba` |
| KNN | `cv_knn_results.csv` | `pred_proba` |
| Linear SVM | `cv_linear_svm_results.csv` | `decision_score` |
| Decision tree | `cv_decision_tree_results.csv` | `pred_proba` |
| Random forest | `cv_random_forest_results.csv` | `pred_proba_class_1` |

SVM decision scores have not been converted into calibrated probabilities. RF/LR
retain their estimator `.predict()` label rule; the probability-threshold rules
in the other original scripts are also retained. Some original prediction CSVs
include feature columns; they have not silently been dropped during packaging.

Per-fold predictions are held-out predictions. The XGBoost script saves these
files, but does not compute a pooled OOF AUC or a pooled OOF SHAP analysis. As in
the original, it saves the final full-data model, not separate fold model files.

Three additional, small run records are written:

- `run_config.json`: the effective settings, with external directory names redacted.
- `environment.json`: observed Python/package versions, not a dump of environment variables.
- `run_metadata.json`: input/code/configuration hashes, run status, and timestamps.

`summary.json` now uses a repository-relative input reference or
`<external>/<filename>`. Redacted external locations must be supplied through CLI
arguments before reusing a settings snapshot. Local console tracebacks can still
contain local paths; the package does not claim to sanitize arbitrary external
logs, datasets, or serialized models.

## Publication boundaries

The seven supplied source files contained no matches for the inspected patterns
for personal absolute paths, passwords/token assignments, GitHub token formats,
private-key blocks, email addresses, or numeric IP addresses. This is a bounded
inspection, not a guarantee that every possible form of sensitive information
has been detected. Obsolete development-history comments were cleaned up.

`.gitignore` excludes local data, results, credentials/key files, model binaries,
and common caches from new tracking. No dataset, local test output, trained model,
or original unedited source file is bundled in this release. No license choice
or authorship information has been invented.

See [docs/CHANGES.md](docs/CHANGES.md) for the source-to-release mapping and
[docs/VERIFICATION.md](docs/VERIFICATION.md) for the actual validation scope.
