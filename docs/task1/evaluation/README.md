# Task 1 evaluation add-on

This is an **overlay** for the existing framework repository. Merge the contents
of `task1_evaluation_github_release/` into the repository root. Do not place that
outer packaging folder inside `scripts/`. No Task 1/Task 2 training file, existing
training configuration, root README, or root dependency file is replaced.

## Scope

The supplied 15 source files were organized as:

- Seven classifier-specific bootstrap entrypoints.
- Four feature-removal entrypoints for the XGBoost ablations.
- Four XGBoost ablation fitting/evaluation entrypoints.

One `_eval_runtime.py` helper and eleven JSON configurations are included. The
seven bootstrap implementations remain separate to retain the source-specific
prediction contracts and validation behavior. Each ablation configuration is
shared by its preparation and fitting entrypoints.

**No executed XGBoost SHAP analysis was present in the 15 supplied files.**
`scripts/evaluation/task1/shap/README.md` records that gap. It is a placeholder,
not an implemented SHAP workflow. No inferred background sample, feature subset,
SHAP explainer or alternative explanation method has been substituted.

No research CSV, model checkpoint, synthetic fixture, console log, raw source
backup, access credential or unpublished result is bundled.

## Layout

```text
scripts/evaluation/task1/
    _eval_runtime.py
    bootstrap/
        bootstrap_ci_xgboost.py
        bootstrap_ci_decision_tree.py
        bootstrap_ci_gaussian_nb.py
        bootstrap_ci_knn.py
        bootstrap_ci_linear_svm.py
        bootstrap_ci_logistic_regression.py
        bootstrap_ci_random_forest.py
    ablation/
        prepare_remove_global.py
        prepare_remove_interface.py
        prepare_remove_contacts.py
        prepare_remove_hbonds.py
        run_xgboost_remove_global.py
        run_xgboost_remove_interface.py
        run_xgboost_remove_contacts.py
        run_xgboost_remove_hbonds.py
    shap/
        README.md                   # source-not-supplied notice only
configs/task1/evaluation/
    bootstrap/                      # seven model JSONs
    ablation/                       # four variant JSONs
        remove_global.json
        remove_interface.json
        remove_contacts.json
        remove_hbonds.json
data/task1/ablation/
    README.md
    .gitignore
docs/task1/evaluation/
    README.md
    README_zh-CN.md
    CHANGES.md
    VERIFICATION.md
    verification_results.json
    file_mapping.json
    requirements.txt
    requirements-tested.txt
```

Ablation scripts fit models, but their role is a controlled feature-removal
experiment; keeping both stages under `evaluation/task1/ablation/` is the chosen
organization. They need not also be copied into `training/`. Configuration files
are data, not Python entrypoints. Keep `_eval_runtime.py` in the location shown.

## Dependencies

An existing training environment may already provide these dependencies. For a
separate environment, the scoped list is:

```bash
python -m pip install -r docs/task1/evaluation/requirements.txt
```

`requirements-tested.txt` records the actual versions used for the included
synthetic verification. It is not the historical environment that produced the
paper. The unpinned list is an inventory, not a claim that every version
combination is supported. There is no new SHAP dependency in this add-on.

## Path and output conventions

JSON paths are relative to the **repository root**. Paths explicitly passed via
`--config`, `--data`, `--result-dir`, `--output-dir` or `--output-file` are relative
to the **terminal working directory**. Absolute paths can be passed at runtime;
none is hard-coded into the released experiment settings.

`--dry-run` reports settings without reading research tables or creating outputs.
For bootstrap, specify the directory of one completed run, not the parent
containing multiple dated runs. No automatic latest-run selection is performed.

A populated output directory or an existing prepared CSV is refused. Use a new
output path for a deliberate re-run. The programs do not delete old results.

## 1. Confidence intervals

For example, when a completed XGBoost run is stored at
`results/task1/xgboost/run_01/`:

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py --result-dir results/task1/xgboost/run_01
```

The entrypoint reads `configs/task1/evaluation/bootstrap/xgboost.json` by default.
Use the corresponding script and result directory for the other classifiers.

| Model JSON / entrypoint suffix | Prediction files | Score column | Source decision rule |
| --- | --- | --- | --- |
| `xgboost` | `fold*_xgb_predictions.csv` | `pred_proba` | `p >= 0.5` |
| `decision_tree` | `fold*_decision_tree_predictions.csv` | `pred_proba` | `p >= 0.5` |
| `gaussian_nb` | `fold*_gaussian_nb_predictions.csv` | `pred_proba` | `p >= 0.5` |
| `knn` | `fold*_knn_predictions.csv` | `pred_proba` | `p >= 0.5` |
| `linear_svm` | `fold*_linear_svm_predictions.csv` | `decision_score` | `score > 0` |
| `logistic_regression` | `fold*_logistic_regression_predictions.csv` | `pred_proba_class_1` | `p > 0.5` |
| `random_forest` | `fold*_random_forest_predictions.csv` | `pred_proba_class_1` | `p > 0.5` |

All prediction inputs use the supplied native `0/1` labels, `protein_id`, and
saved `pred_label` where required by the original script. Do not rename SVM
scores to probabilities. The six non-XGBoost programs also require their matching
`cv_<model>_results.csv` with the original per-fold metrics. Prediction and CV
files must come from the same run.

Defaults remain 2000 bootstrap repetitions, seed 42, confidence 0.95, and
maximum redraw setting 10000. Sampling is by `protein_id` **within each outer
fold**, retaining all frames of the sampled protein and any mixed within-protein
labels. Each replicate averages the fold metrics; it is not a pooled OOF AUC or
an average of per-protein AUCs. The confidence interval is the percentile interval
of these fold-mean replicates. The models are not retrained.

The original validation differences are retained:

- XGBoost reconstructs labels using `p >= 0.5` and warns about saved-label
  mismatches. Its CV comparison is optional/warning-only. It discovers the
  available folds rather than adding a new mandatory-five-fold check; use all
  five paper folds for the reported Task 1 analysis.
- The other six require folds 1–5, require their CV table, and stop on the
  original prediction-rule/CV consistency failures. The original CV tolerances
  are retained: KNN `1e-5`, LR `1e-7`, and DT/NB/SVM/RF `1e-8`.

These differences have not been silently unified. The original label validators
and invalid-input behavior have not been broadly redesigned.

An explicit output directory and command-line parameter override can be used:

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_random_forest.py --result-dir results/task1/random_forest/run_01 --output-dir results/task1/evaluation/random_forest_ci_run_01 --n-bootstrap 2000 --seed 42
```

By default, output is written to `<result-dir>/bootstrap/`:

```text
bootstrap_ci_summary.csv
bootstrap_distribution.csv
recomputed_fold_metrics.csv
protein_label_composition.csv
bootstrap_metadata.json
run_config.json
environment.json
run_metadata.json
```

The first five names and their numerical content conventions are retained.
The last three document successful execution of the publication interface.
The reported intervals describe resampling uncertainty of **fixed outer-test
predictions**, not uncertainty from repeating the entire tuning/training process.

## 2. XGBoost ablations

Default full input:

```text
data/task1/proid_0_1_feature_labels_standard.csv
```

The original positional format remains: first column is the group identifier,
last column is the label, and every intermediate column is a model feature.
No extra `frame_id`/index column is automatically excluded. Existing feature
values are used without additional scaling, imputation or label merging.

### Prepare the reduced tables

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py
python scripts/evaluation/task1/ablation/prepare_remove_interface.py
python scripts/evaluation/task1/ablation/prepare_remove_contacts.py
python scripts/evaluation/task1/ablation/prepare_remove_hbonds.py
```

| Variant | Removal rule from source | Expected removed features | Prepared CSV |
| --- | --- | ---: | --- |
| `remove_global` | Eight explicitly named whole-protein scalar columns | 8 | `data/task1/ablation/removeW.csv` |
| `remove_interface` | Five explicitly named interfacial scalar columns | 5 | `data/task1/ablation/remove_Inter.csv` |
| `remove_contacts` | `^[A-Z]{3}-[A-Z]{3}$` | 210 | `data/task1/ablation/remove_resn.csv` |
| `remove_hbonds` | `^hbond[A-Z]{3}-[A-Z]{3}$` | 210 | `data/task1/ablation/remove_hbond.csv` |

The exact column names are in the corresponding JSON, not guessed from shortened
paper display names. `remove_global` corresponds to the old `remove_W` script;
it removes eight whole-protein scalar features, not only water-related features.

For regex groups, finding no matching columns remains an error. A count other
than 210 remains a warning, not a newly imposed hard failure. Preparation retains
row order, remaining column order and labels. It writes a new CSV and a
`<CSV>.metadata.json` sidecar listing the deleted columns.

Two legacy preparation output names did not match their training inputs:
`removeW_1_2_feature_labels_standard.csv` and
`removeInter_1_2_feature_labels_standard.csv`. This release aligns them with
`removeW.csv` and `remove_Inter.csv`. **No 1/2-to-0/1 conversion is performed.**

An existing reduced dataset can be used directly for fitting; regenerating it is
not necessary. To change preparation locations:

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py --data data/task1/proid_0_1_feature_labels_standard.csv --output-file data/task1/ablation/removeW_v2.csv
```

### Fit and evaluate each ablation

```bash
python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_interface.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_contacts.py --n-jobs -1
python scripts/evaluation/task1/ablation/run_xgboost_remove_hbonds.py --n-jobs -1
```

`--data` for these fitting entrypoints means an **already reduced table**. They
do not delete the feature group a second time. Do not point them to the full
feature matrix expecting automatic removal.

Default result directories are
`results/task1/ablation/<variant>/<timestamp>/`. An explicit example is:

```bash
python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py --data data/task1/ablation/removeW.csv --output-dir results/task1/ablation/remove_global/run_01 --n-jobs -1
```

The original default protocol remains: five outer GroupKFold folds, one
group-aware inner holdout per fold, 50 candidate parameter settings, 3000 boosting
iterations as the upper bound, and 100-round early stopping. Candidate generation
order, class weighting, feature filtering, fitting subsets, final full-data
model logic, prediction columns and result filenames are retained. This is the
source's per-ablation retuning workflow, not an alternative fixed-hyperparameter
ablation. `--n-jobs -1` does not distribute folds across cluster nodes.

The configured `BASELINE_AUC = 0.7931` is retained as the source's historical
**display-only reference**. It is not a verified Task 1 full-feature baseline
from this release and is not used for fitting/selection. Setting it to JSON
`null` suppresses that comparison without changing the training protocol.

To compute an ablation's confidence interval, reuse the XGBoost bootstrap script
on that ablation's completed result directory:

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py --result-dir results/task1/ablation/remove_global/run_01
```

## Privacy and verification limits

The successful-run JSONs redact external directory prefixes, retaining filenames
and scientific column names. No hostname, environment-variable dump, SSH key or
console log is automatically recorded. Console output and exception tracebacks
can still include local paths; this helper is not a universal log sanitizer.
The scoped sensitive-pattern scan found no matches in the 15 supplied scripts,
16 released Python files and 11 configurations. It does not cover unprovided
files, repository history or the contents of research data.

Local Git ignore rules do not prevent a deliberate web upload or remove files
already tracked. No actual dataset is present in this add-on.

See `CHANGES.md`, `VERIFICATION.md`, `verification_results.json`, and
`file_mapping.json`. All seven bootstrap programs and all four preparation/fitting
pairs passed the documented synthetic comparisons. No real Task 1 dataset was
provided, no paper AUC/CI was verified, and no SHAP analysis was executed.
