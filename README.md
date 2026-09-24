# IL-domain-framework

## Molecular dynamics and grouped machine learning for ionic-liquid effects on two-domain proteins

[中文版](README_zh-CN.md) · [Repository](https://github.com/YuZhang-07/IL-domain-framework) · [Data releases](https://github.com/YuZhang-07/IL-domain-framework/releases)

This repository contains research code, configuration files, and selected molecular dynamics (MD) inputs for studying ionic-liquid effects on two-domain protein interfaces. The machine-learning workflows operate on precomputed frame-level descriptors and implement two binary classification tasks, with protein systems kept separate during grouped model evaluation.

The repository brings together **Task 1 and Task 2 classifier training, protein-cluster bootstrap confidence intervals, XGBoost feature-group ablation, and final-model refitting for subsequent SHAP interpretation**. The unstandardized feature-and-label dataset is distributed separately as a release asset. The MD directory provides representative inputs and an execution script; raw simulation trajectories are not included.

This is the project-level guide. Component README, change, and verification files describe their respective modules and the scope of their original code packages. Statements in those component files that a particular code package contains no dataset do not mean that the separately distributed dataset is unavailable.

[Data](#data) · [Input format](#inputs) · [Environment](#environment) · [Training](#training) · [Bootstrap](#bootstrap) · [Ablation](#ablation) · [SHAP model preparation](#shap) · [MD](#md) · [Scope](#scope) · [License and citation](#license)

## 1. Repository layout

```text
IL-domain-framework/
├── README.md
├── README_zh-CN.md
├── requirements.txt
├── requirements-tested.txt
├── .gitignore
├── .gitattributes
├── metadata/                         # Dataset information and local master-table location
├── data/
│   ├── task1/                        # Prepared Task 1 input; local ablation tables
│   └── task2/                        # Prepared Task 2 input; local ablation tables
├── configs/
│   ├── task1/                        # Classifier JSON files
│   │   └── evaluation/               # bootstrap/, ablation/, shap/
│   └── task2/
│       └── evaluation/
├── scripts/
│   ├── training/
│   │   ├── task1/                    # Seven classifiers and _runtime.py
│   │   └── task2/                    # Seven classifiers and _runtime.py
│   └── evaluation/
│       ├── task1/                    # _eval_runtime.py; bootstrap/, ablation/, shap/
│       └── task2/                    # _eval_runtime.py; bootstrap/, ablation/, shap/
├── MD/
│   ├── scripts/run_md.sh
│   ├── input_templates/              # amber/, packmol/, tleap/
│   └── structures/Bmim.pdb
└── docs/                             # Component instructions and verification records
```

`results/` is created locally by the analysis programs; it is not a promise that precomputed manuscript results are included in the source download. The large `feature_labels.csv` file is obtained separately and placed at the local path described below.

Keep the directory hierarchy intact. Each task's training scripts need their own `_runtime.py`; Bootstrap and ablation need their task's `_eval_runtime.py`. Do not interchange helpers between tasks or move individual scripts away from their configuration tree. The SHAP refit entry points are independent of these helpers but still need their corresponding JSON configuration.

Paths in this guide use the repository's **uppercase `MD/`** directory. When consulting older component examples that use `md/`, use `MD/` for this checkout.

<a id="data"></a>
## 2. Obtain the feature-and-label dataset

Open [Data releases](https://github.com/YuZhang-07/IL-domain-framework/releases), select the release containing the dataset, and download **`feature_labels.csv`** from **Assets**. The initial data release is titled **Workflow data v1.0.0**; use the actual release tag and URL shown on its page when identifying a version.

| Item | Description |
|---|---|
| Dataset filename | `feature_labels.csv` |
| Contents | Full, unstandardized feature data and associated labels |
| Format | CSV |
| Local destination after download | `metadata/feature_labels.csv` |
| Relationship to training inputs | Master feature table; not a ready-made standardized input for either task |

Retain the filename and place the file under the repository root as follows:

```text
metadata/
└── feature_labels.csv
```

The dataset is a table of **derived features**, not a raw MD trajectory archive. “Unstandardized” describes the feature scaling state; it does not mean that no feature extraction or other upstream processing has occurred.

Download the data asset separately from the source code. GitHub's automatically generated `Source code (zip)` and `Source code (tar.gz)` contain the repository snapshot, not separately attached release assets. See [GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Use matching code and data versions. A later commit on `main` is not automatically part of an earlier release tag. Repository and release access depend on the repository's visibility and permissions; this README does not itself make a private repository public.

<a id="inputs"></a>
## 3. Task inputs and label handling

The training entry points expect **task-specific, already prepared feature tables**, with the following default local paths:

| Task | Default input path | Input labels | Internal encoding | Positive class in saved prediction scores |
|---|---|---|---|---|
| Task 1 | `data/task1/proid_0_1_feature_labels_standard.csv` | `0`, `1` | `0 → 0`, `1 → 1` | Input label `1` |
| Task 2 | `data/task2/proid_1_2_feature_labels_standard.csv` | `1`, `2` | `1 → 0`, `2 → 1` | Original label `2` |

These paths specify **required local inputs**, not additional assets that are necessarily included in the data release. An input in another location can be supplied with `--data`.

### Master table versus prepared task tables

The training entry points do not construct these task tables from `metadata/feature_labels.csv`. Task-specific sample selection, scientific label assignment, and feature standardization are upstream steps. Exact replication requires the original task preparation rules and transformations, or the matching prepared tables. This guide does not introduce a substitute preprocessing recipe.

**Do not simply rename the unstandardized master table to one of the standardized training filenames.** The programs use the supplied feature values without adding a scaler. Task 1 does not automatically combine a third label with label `1`; Task 2 performs only the stated `1/2 → 0/1` encoding. Biological response definitions and sample inclusion must follow the actual study annotation, not be inferred from integer labels alone.

### Positional CSV schema

```text
protein_id,feature_1,feature_2,...,feature_N,label
```

The **first column** supplies protein grouping IDs, the **last column** supplies task labels, and **every intermediate column** is treated as a model feature. Preserve column names, feature order, and the original sample order when reproducing a run. Do not insert `frame_id`, an exported dataframe index, or other non-predictor metadata among the feature columns.

Labels are frame-level and can differ within the same `protein_id`. The workflows do not collapse a protein to a single class. Task 2 predictions retain original labels `1/2` in `true_label` and `pred_label`.

The retained training behavior permits missing feature values in XGBoost; the six other classifier scripts reject them. Infinite feature values are rejected. No additional imputation is performed. Grouped splitting within these scripts does not, by itself, establish how upstream standardization was fitted.

<a id="environment"></a>
## 4. Python environment

Run the commands in this guide from the **repository root**. Multiline commands use Bash continuation syntax; in Windows PowerShell or Command Prompt, join the lines into a single command and remove the trailing `\` characters.

Install the shared Python dependencies into your chosen environment:

```bash
python -m pip install -r requirements.txt
```

Alternatively, request the direct package versions recorded during the code-packaging comparisons:

```bash
python -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` records comparisons performed on Python 3.13.5. It is **not a reconstruction of the original manuscript environment**, a complete transitive lockfile, or evidence that all results have been reproduced on every platform. Use the original research environment where exact historical replication is required and it is available. Module-specific dependency records are included in the corresponding documentation directories.

The SHAP model-refitting scripts do not require the `shap` package because they do not calculate SHAP values. The separate, optional logistic-regression SHAP branch is disabled by default and is not the XGBoost interpretation workflow. MD requires a separate local installation of AMBER and, for system construction, Packmol and `tleap`; these executables are not installed by the Python requirements files.

<a id="training"></a>
## 5. Train the classifiers

### Classifiers and configurations

The following names apply within both `scripts/training/task1/` and `scripts/training/task2/`. Each program automatically reads its matching JSON under `configs/task1/` or `configs/task2/`.

| Classifier | Training script | Configuration filename |
|---|---|---|
| XGBoost | `train_xgboost.py` | `xgboost.json` |
| Logistic regression | `train_logistic_regression.py` | `logistic_regression.json` |
| Gaussian naive Bayes | `train_gaussian_nb.py` | `gaussian_nb.json` |
| K-nearest neighbors | `train_knn.py` | `knn.json` |
| Linear SVM | `train_linear_svm.py` | `linear_svm.json` |
| Decision tree | `train_decision_tree.py` | `decision_tree.json` |
| Random forest | `train_random_forest.py` | `random_forest.json` |

With the prepared task inputs available, run XGBoost as follows.

**Task 1**

```bash
python scripts/training/task1/train_xgboost.py \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --output-dir results/task1/xgboost/run_01 \
  --n-jobs -1
```

**Task 2**

```bash
python scripts/training/task2/train_xgboost.py \
  --data data/task2/proid_1_2_feature_labels_standard.csv \
  --output-dir results/task2/xgboost/run_01 \
  --n-jobs -1
```

Use the corresponding script for another classifier. `run_01` is an example run name; keep each task, classifier, and run in a separate output directory. Omit `--output-dir` to obtain an automatically timestamped directory.

### Evaluation protocol

The default training configuration uses **five outer `GroupKFold` folds** with `protein_id` as the group, random seed **42**, and **50 requested candidate parameter settings** per outer fold. Candidates are generated by the model-specific sampling or enumeration code, not by Optuna.

Within each outer-training fold, model selection uses **one group-aware inner training/validation holdout**, constructed using the original `StratifiedGroupKFold` / `GroupShuffleSplit` fallback and retry logic. This is not a five-fold inner average. Hyperparameters are selected by inner-validation AUC, and XGBoost also uses that validation subset for early stopping. The outer test subset is not used for these selections.

The evaluated fold model is fitted on **inner-training samples**, not refitted on the entire outer-training fold. Near-constant feature filtering remains based on the outer-training fold. The separately saved full-data model does not replace held-out evaluation. Report the outer-fold metrics from the matching CV table; do not substitute the full-data model's training performance.

### Paths and command-line options

Relative paths stored **inside JSON** are resolved from the repository root. Explicit relative paths supplied on the **command line** are resolved from the shell's current working directory. Local absolute paths may be supplied at runtime without embedding them into the shared source files.

Training entry points support `--config`, `--data`, `--output-dir`, `--n-jobs`, `--n-trials`, `--seed`, and `--dry-run`. A custom configuration must be a complete configuration for the selected model. Changing seeds, candidate counts, search spaces, or validation settings changes the experiment.

```bash
python scripts/training/task1/train_xgboost.py --help
python scripts/training/task2/train_xgboost.py --dry-run
```

For these training entry points, `--dry-run` prints the effective configuration and input-file existence without reading the dataset or fitting models. It is not a dataset-content validation. `--n-jobs -1` retains supported estimator parallelism; it does not distribute folds over cluster nodes or add model-level parallelism to estimators without that interface.

<a id="bootstrap"></a>
## 6. Bootstrap confidence intervals

Bootstrap programs are under `scripts/evaluation/task1/bootstrap/` and `scripts/evaluation/task2/bootstrap/`; configurations are under the corresponding `configs/task*/evaluation/bootstrap/` directories.

Use the saved **outer-test predictions from one completed training run**:

```bash
python scripts/evaluation/task1/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task1/xgboost/run_01

python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/xgboost/run_01
```

Default outputs are written to `<result-dir>/bootstrap/`. The other entry points are `bootstrap_ci_logistic_regression.py`, `bootstrap_ci_gaussian_nb.py`, `bootstrap_ci_knn.py`, `bootstrap_ci_linear_svm.py`, `bootstrap_ci_decision_tree.py`, and `bootstrap_ci_random_forest.py`.

### Prediction columns

All inputs need `protein_id` and `true_label`, plus the model's score column. The six non-XGBoost programs also require saved `pred_label` and the matching `cv_*_results.csv`. XGBoost retains its original optional-label and warning-based CV comparison behavior. Use all five outer folds to reproduce the five-fold analysis, and do not mix runs or tasks.

| Model | CV table | Task 1 score column | Task 2 score column |
|---|---|---|---|
| XGBoost | `cv_xgb_results.csv` | `pred_proba` | `pred_proba` |
| Logistic regression | `cv_logistic_regression_results.csv` | `pred_proba_class_1` | `pred_proba_class_2` |
| Gaussian NB | `cv_gaussian_nb_results.csv` | `pred_proba` | `pred_proba` |
| KNN | `cv_knn_results.csv` | `pred_proba` | `pred_proba` |
| Linear SVM | `cv_linear_svm_results.csv` | `decision_score` | `decision_score` |
| Decision tree | `cv_decision_tree_results.csv` | `pred_proba` | `pred_proba` |
| Random forest | `cv_random_forest_results.csv` | `pred_proba_class_1` | `pred_proba_class_2` |

SVM decision scores are not probabilities. XGBoost, NB, KNN, and DT use the retained `>= 0.5` probability threshold; LR and RF retain their `> 0.5` positive-class convention; SVM uses a decision score `> 0`. Exact ties therefore need not be classified identically across models.

### Statistical interpretation

The default is **2,000 protein-cluster bootstrap replicates**, seed **42**, and a **95% percentile confidence interval**. Protein systems are sampled with replacement **within each outer fold**; all frames of a sampled protein share its bootstrap multiplicity. Each replicate evaluates the folds separately and then averages their metrics.

These intervals describe uncertainty of **fixed outer-test predictions under protein-system resampling**. They are not independent-frame bootstrap intervals, pooled-prediction AUC intervals, or intervals including repeated model fitting and hyperparameter selection.

Main outputs are `bootstrap_ci_summary.csv`, `bootstrap_distribution.csv`, `recomputed_fold_metrics.csv`, `protein_label_composition.csv`, and `bootstrap_metadata.json`, together with run records. See the [Task 1 evaluation guide](docs/task1/evaluation/README.md) and [Task 2 evaluation guide](docs/task2/evaluation/README.md).

<a id="ablation"></a>
## 7. XGBoost feature-group ablation

Ablation consists of **preparing a feature-reduced table and then retraining XGBoost**. Each task has its own programs and configurations. Exact deletion column names or matching expressions are stored in `configs/task*/evaluation/ablation/`.

| Variant | Removed feature group | Configured expected count | Prepared CSV filename |
|---|---|---:|---|
| `remove_global` | Whole-protein scalar descriptors | 8 | `removeW.csv` |
| `remove_interface` | Interfacial scalar descriptors | 5 | `remove_Inter.csv` |
| `remove_contacts` | Interfacial residue-pair contacts | 210 | `remove_resn.csv` |
| `remove_hbonds` | Interfacial residue-pair hydrogen bonds | 210 | `remove_hbond.csv` |

These are configured feature-group sizes, not an independent count of every downloaded data file. Prepared CSVs are placed under `data/task1/ablation/` or `data/task2/ablation/`.

**Task 1 example: remove whole-protein scalar descriptors**

```bash
python scripts/evaluation/task1/ablation/prepare_remove_global.py

python scripts/evaluation/task1/ablation/run_xgboost_remove_global.py \
  --data data/task1/ablation/removeW.csv \
  --output-dir results/task1/ablation/remove_global/run_01 \
  --n-jobs -1
```

**Task 2 example: the corresponding ablation**

```bash
python scripts/evaluation/task2/ablation/prepare_remove_global.py

python scripts/evaluation/task2/ablation/run_xgboost_remove_global.py \
  --data data/task2/ablation/removeW.csv \
  --output-dir results/task2/ablation/remove_global/run_01 \
  --n-jobs -1
```

Replace the variant in the script names and output directory, and use the corresponding CSV from the table for another group. Preparation reads the **complete prepared task input**; the training entry point reads an **already ablated table** and does not delete columns itself. Existing ablated inputs can be supplied directly without repeating preparation.

Each variant retains its own inner-validation parameter search and outer grouped evaluation; this is not a fixed-hyperparameter ablation of one already trained model. Compare it with the matching all-feature run for the same task and sample set. Any display-only historical `BASELINE_AUC` retained in configuration is not a substitute for that all-feature result.

Reuse the XGBoost Bootstrap program for ablation outputs, for example:

```bash
python scripts/evaluation/task2/bootstrap/bootstrap_ci_xgboost.py \
  --result-dir results/task2/ablation/remove_global/run_01
```

<a id="shap"></a>
## 8. Final XGBoost model preparation for SHAP

The entry points in `scripts/evaluation/task*/shap/retrain_xgboost_for_shap.py` **prepare a final model for subsequent interpretation; they do not calculate SHAP values or create SHAP plots**.

They read the matching prepared task table and an existing `cv_xgb_results.csv`, deduplicate its per-fold `best_params`, and evaluate those candidates using full-data internal group-aware cross-validation. The selected candidate's median `best_iteration + 1` determines the final tree count. Early stopping is then disabled for the final full-data fit.

This model-preparation step evaluates all internal folds, unlike the single inner holdout used within each outer fold during classifier training. Its internal AUC is a selection record, **not a new estimate of generalization performance**. The original outer-CV file is read-only and remains the source of the reported outer-fold AUC.

```bash
python scripts/evaluation/task1/shap/retrain_xgboost_for_shap.py \
  --data data/task1/proid_0_1_feature_labels_standard.csv \
  --cv-results results/task1/xgboost/run_01/cv_xgb_results.csv \
  --output-dir results/task1/shap/final_model_for_shap/run_01 \
  --n-jobs -1

python scripts/evaluation/task2/shap/retrain_xgboost_for_shap.py \
  --data data/task2/proid_1_2_feature_labels_standard.csv \
  --cv-results results/task2/xgboost/run_01/cv_xgb_results.csv \
  --output-dir results/task2/shap/final_model_for_shap/run_01 \
  --n-jobs -1
```

Each task reads `configs/task*/evaluation/shap/refit_xgboost.json`. The CV file must be explicitly supplied; the scripts do not select the newest run automatically.

Main outputs are `xgb_final_model.joblib`, `final_model_for_shap_meta.json`, `candidate_parameter_evaluation.csv`, and `candidate_fold_details.csv`, plus run records. `features_used_by_model` records features used in tree splits, not SHAP importance. No fixed Top17 or Top20 selection is introduced.

The refit programs alone are not a complete release of the numerical SHAP analysis. Background sampling, explanation settings, SHAP-value computation, and figure generation require the actual interpretation implementation. See [the SHAP model-preparation guide](docs/shap_model_refit/README.md).

<a id="md"></a>
## 9. Representative MD inputs and execution

The [MD directory](MD/) contains selected input templates and an execution script:

| Location | Contents |
|---|---|
| `MD/input_templates/amber/` | `min1.in`, `min2.in`, `min3.in`, `heat.in`, `eq1.in`, `eq2.in`, `eq3.in` |
| `MD/input_templates/packmol/packmol_example.inp` | One representative Packmol construction input |
| `MD/input_templates/tleap/tleap_example.in` | Representative topology-preparation input |
| `MD/structures/Bmim.pdb` | Bmim molecular coordinates |
| `MD/scripts/run_md.sh` | Ordered AMBER execution and restart-file handoff |

**The Packmol input is an example, not the complete set of protein-specific construction settings.** Box dimensions varied between protein systems. The example packing bounds, periodic box, and restraint residue masks must not be treated as universal values for every protein.

The original sequence is retained: three minimizations, heating, four successive uses of `eq3.in`, one use of `eq1.in`, and ten successive uses of `eq2.in`.

This directory does not include raw trajectories, all protein starting structures, a full topology collection, or the complete Bmim parameterization. The construction examples reference local files such as `protein.pdb`, `water.pdb`, `Bmim.mol2`, and `Bmim.frcmod`; the MD runner requires a valid prepared topology and initial coordinates. `Bmim.pdb` does not replace force-field parameters.

```bash
bash MD/scripts/run_md.sh --help
```

With valid local `cmp.prmtop` and `cmp.inpcrd` already prepared in the example working directory:

```bash
bash MD/scripts/run_md.sh \
  --work-dir MD/work/system_example \
  --output-dir MD/work/system_example/md_output
```

Read [the MD guide](MD/README.md), including its recorded input limitations, before execution. In particular, it documents the heating-input `&wt` / `nmropt` issue rather than silently replacing the original protocol. These files are representative workflow materials, not an independently runnable archive of all simulations. Omitting trajectories from the repository does not disable trajectory writing in local MD runs.

## 10. Outputs and run records

Typical local output locations are:

```text
results/task1/<classifier>/<run>/
results/task2/<classifier>/<run>/
results/task1/ablation/<variant>/<run>/
results/task2/ablation/<variant>/<run>/
results/task1/shap/final_model_for_shap/<run>/
results/task2/shap/final_model_for_shap/<run>/
```

Bootstrap uses the selected training run's `bootstrap/` subdirectory unless another output path is specified. The programs reject populated output directories; ablation preparation rejects an existing output table. They do not delete previous results to make room for a new run.

Training and evaluation write `run_config.json`, `environment.json`, and `run_metadata.json` to identify effective settings, observed software versions, and input/code references. Record fields vary by module; see the corresponding documentation. These files do not turn synthetic verification runs into manuscript results.

Shared JSON records redact supported external directory references as `<external>/<filename>`. Such records are not always reusable as input configurations without restoring local paths. Console tracebacks and MD-engine logs can still contain local paths. The ignore files are local Git rules, not a guarantee that arbitrary uploaded files have been sanitized.

<a id="scope"></a>
## 11. Reproducibility scope and component documentation

The code supports classifier fitting from prepared task tables, evaluation from matching held-out predictions, ablation retraining, and final-model preparation for interpretation. The release asset additionally provides the unstandardized master feature table. These are distinct inputs and stages.

The supplied training entry points do not reconstruct the master table from trajectories or define a complete conversion from that table to the original standardized task inputs. The SHAP refit programs do not implement SHAP computation. The selected MD inputs omit trajectories and external system dependencies. Consequently, this release should not be interpreted as a one-command, end-to-end reproduction of every manuscript result from raw simulation data.

Exact reproduction also depends on matching the data version, sample selection, preprocessing, software environment, and experiment configuration. Program-generated outputs must be distinguished from precomputed results actually attached to a release. No manuscript AUC, confidence interval, DOI, or publication status is inferred from example filenames or documentation.

| Component | Usage documentation | Recorded verification scope |
|---|---|---|
| Task 1 training | This root guide and [input notes](data/task1/README.md) | [Training verification](docs/VERIFICATION.md) |
| Task 2 training | [Task 2 guide](docs/task2/README.md) | [Task 2 verification](docs/task2/VERIFICATION.md) |
| Task 1 Bootstrap and ablation | [Task 1 evaluation guide](docs/task1/evaluation/README.md) | [Task 1 evaluation verification](docs/task1/evaluation/VERIFICATION.md) |
| Task 2 Bootstrap and ablation | [Task 2 evaluation guide](docs/task2/evaluation/README.md) | [Task 2 evaluation verification](docs/task2/evaluation/VERIFICATION.md) |
| SHAP model preparation | [Refit guide](docs/shap_model_refit/README.md) | [Refit verification](docs/shap_model_refit/VERIFICATION.md) |
| MD example inputs | [MD guide](MD/README.md) | Checks and limitations documented in the same guide |

The recorded code comparisons used the limited test conditions described in those files. They are not independent validation of the full scientific dataset or the paper's conclusions; the MD packaging checks did not execute real AMBER or Packmol simulations.

<a id="license"></a>
## 12. Licensing and citation

### Reuse terms

This README does **not** assign a software or dataset license. Consult the explicit license files and the license statement accompanying the specific data release for the applicable terms. If no license has been provided for a material, do not infer an open-source or open-data grant from its visibility or download availability. Third-party software and materials retain their own terms. See [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

### Cite the version used

When referencing this workflow, identify **IL-domain-framework**, the repository URL, and the exact code tag or commit used. When referencing the dataset, additionally identify **`feature_labels.csv`** and its version-specific release URL. Use the actual bibliographic metadata of the associated paper or an archived software/data record when those records are available; no manuscript citation or archive DOI is supplied here without a verified record.

The [releases page](https://github.com/YuZhang-07/IL-domain-framework/releases) is the dataset discovery entry point. For a paper's data citation, copy the individual release URL rather than citing only the moving `main` branch or `latest` release. A release title and its Git tag are not necessarily identical.

## 13. Questions and implementation issues

Use [repository Issues](https://github.com/YuZhang-07/IL-domain-framework/issues) for questions about the shared implementation, when accessible. Include the task, script, code version, relevant configuration, and error message. Do not post credentials, private system paths, or data that are outside the intended public release.
