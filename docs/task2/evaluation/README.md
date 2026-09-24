# Task 2 evaluation: bootstrap confidence intervals and XGBoost ablation

This is an additive release based on 15 supplied scripts: seven classifier
bootstrap programs, four feature-removal programs and four ablation fitting
programs. **No XGBoost SHAP implementation was supplied.** The `shap/README.md`
records that missing component and is not executable analysis code.

The Chinese guide is [README_zh-CN.md](README_zh-CN.md). Exact source-to-release
filenames and hashes are in [file_mapping.json](file_mapping.json).

## Installation and layout

Merge the contents inside `task2_evaluation_github_release/` into the existing
repository root, preserving existing files. Do not replace whole existing
folders. No root README, training code, or Task 1 evaluation file is replaced.

```text
scripts/evaluation/task2/_eval_runtime.py
scripts/evaluation/task2/bootstrap/bootstrap_ci_*.py
scripts/evaluation/task2/ablation/prepare_remove_*.py
scripts/evaluation/task2/ablation/run_xgboost_remove_*.py
scripts/evaluation/task2/shap/README.md
configs/task2/evaluation/bootstrap/*.json
configs/task2/evaluation/ablation/*.json
data/task2/ablation/
docs/task2/evaluation/
```

Keep `_eval_runtime.py` with the scripts. It implements configuration, path and
record-keeping interfaces, not statistical or ML kernels. It has no dependency
on Task 1. Use an existing environment compatible with the original scripts or:

```bash
python -m pip install -r docs/task2/evaluation/requirements.txt
```

The tested versions are recorded in `requirements-tested.txt`. These are the
observed packaging-test versions, not an inferred historical paper environment,
and not a complete transitive dependency lock. Validation scope is described in
[VERIFICATION.md](VERIFICATION.md).

## Path rules

JSON-relative paths resolve against the repository root. Explicit CLI-relative
paths resolve against the current working directory. Bootstrap requires a
specific completed run via `--result-dir` or `input.result_dir` in its JSON; it
never guesses the latest run. Output directories must be new/empty. Prepared
CSV files must not exist. Raw research inputs are never modified.

Each entrypoint supports `--help`, `--config` and `--dry-run`. The run records
are descriptive outputs; use the input JSON schema under `configs/` for reruns.

## Bootstrap

From the repository root; replace `run_01` with the actual completed run:

```bash
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/xgboost/run_01
```

By default, outputs go to `<result-dir>/bootstrap/`. `--output-dir` selects a
new destination. Defaults remain 2000 resamples, seed 42, confidence 0.95 and
10000 allowed single-class redraws. Overrides include `--n-bootstrap`, `--seed`,
`--confidence`, and `--max-redraw`.

Input labels are original **1/2** and are explicitly mapped as 1 -> 0, 2 -> 1;
**original label 2 is positive**. Do not convert input predictions to Task 1
labels. A cluster may contain frames of both labels.

| Classifier | Prediction suffix | Score column | Original-label-2 rule |
| --- | --- | --- | --- |
| XGBoost | `fold*_xgb_predictions.csv` | `pred_proba` | `>= 0.5` |
| Decision tree | `fold*_decision_tree_predictions.csv` | `pred_proba` | `>= 0.5` |
| Gaussian NB | `fold*_gaussian_nb_predictions.csv` | `pred_proba` | `>= 0.5` |
| KNN | `fold*_knn_predictions.csv` | `pred_proba` | `>= 0.5` |
| Linear SVM | `fold*_linear_svm_predictions.csv` | `decision_score` | `> 0` |
| Logistic regression | `fold*_logistic_regression_predictions.csv` | `pred_proba_class_2` | `> 0.5` |
| Random forest | `fold*_random_forest_predictions.csv` | `pred_proba_class_2` | `> 0.5` |

Use the corresponding `bootstrap_ci_<classifier>.py` entrypoint. All tables
need `protein_id` and `true_label`. `pred_label` is mandatory except for XGBoost,
whose original optional/warning-based handling is preserved. Six non-XGBoost
programs require the matching five-fold `cv_*_results.csv` and strictly validate
it. XGBoost retains its original file discovery and warning-based CV comparison;
provide all five folds to reproduce the paper's five-fold statistic.

Resampling is within each outer fold at the `protein_id` cluster level. A
replicate evaluates every fold then averages its metrics; intervals use the
percentile method. No frame-independent resampling, group-label stratification,
pooled-OOF AUC or refitting has been substituted. The intervals describe fixed
prediction uncertainty under protein-cluster resampling, not full model
retraining or hyperparameter-search uncertainty. SVM scores are not probabilities.

Task 2 tolerances remain source-specific: LR `1e-7`; DT/NB/KNN/SVM/RF `1e-8`;
XGBoost retains its original `1e-8` warning threshold. Task 1 KNN's tolerance is
not copied into Task 2.

Original CSV/metadata filenames are preserved. Added successful-run records:
`run_config.json`, `environment.json`, `run_metadata.json`.

## XGBoost ablation

Default full input: `data/task2/proid_1_2_feature_labels_standard.csv`.
Column layout is unchanged: first column = group, final column = label, every
intermediate column = numeric feature. No rescaling, imputation or relabeling is
added. Do not insert non-feature index/frame columns into the feature block.

| Variant | Deleted group | Prepared input |
| --- | --- | --- |
| `remove_global` | Eight explicitly named global scalars | `removeW.csv` |
| `remove_interface` | Five explicitly named interface scalars | `remove_Inter.csv` |
| `remove_contacts` | `^[A-Z]{3}-[A-Z]{3}$`, expected 210 | `remove_resn.csv` |
| `remove_hbonds` | `^hbond[A-Z]{3}-[A-Z]{3}$`, expected 210 | `remove_hbond.csv` |

Prepared inputs live in `data/task2/ablation/`; each variant has one JSON shared
by preparation and fitting. Exact explicit feature names and regexes come from
the supplied sources. Regex count mismatches remain warnings, zero matches remain
errors. The original global/interface preparation basenames have been explicitly
aligned with their original fitting inputs; see `CHANGES.md`.

```bash
python scripts/evaluation/task2/ablation/prepare_remove_global.py
python scripts/evaluation/task2/ablation/run_xgboost_remove_global.py \
  --output-dir results/task2/ablation/remove_global/run_01 --n-jobs -1
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/ablation/remove_global/run_01
```

Use the analogous `remove_interface`, `remove_contacts`, or `remove_hbonds`
entrypoints for other variants. Existing ablated data can be passed directly
with `--data`; skip preparation. **The fitting entrypoint expects an already
ablated table and does not delete columns again.** Preparation also accepts
`--data` (full input) and `--output-file`. Default fitting outputs are timestamped
under `results/task2/ablation/<variant>/`.

Default five outer folds, 50 candidates, 3000-tree cap, 100-round early stopping,
group-aware inner validation, class weighting, near-constant handling, candidate
order and final full-data model behavior are preserved. Each ablation still
retunes independently. Outer evaluation retains the original inner-training
fitting subset; full-data final models do not replace outer-CV metrics.

Historical `BASELINE_AUC = 0.7931` is preserved only as the source's display
reference and is not independently verified. It does not affect fitting or
selection. JSON `null` disables this comparison without changing model fitting.
Do not assume it is the manuscript's final full-feature baseline.

## Privacy and scope

No research CSV, trained model, original script duplicate, server log or Git
history is included. External absolute directory prefixes in generated shareable
JSON records are redacted; basenames and scientific feature names are retained.
**Console logs, tracebacks and existing arbitrary files are not automatically
sanitized.** No complete environment-variable dump, username or hostname is
collected. The limited pattern review is not a security guarantee for unrelated
files, research data, or repository history.

The four preparation CSVs are locally ignored by the supplied `.gitignore`.
This does not replace the manuscript's research-data archival/access policy.

SHAP must be added from the actual manuscript workflow; saving a final model
for possible SHAP use does not constitute a SHAP implementation.
