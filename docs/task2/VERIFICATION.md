# Task 2 verification record

Verification date (UTC): 2026-09-23
Python used: 3.13.5

## Purpose and bounded scope

These checks address the publication refactor's main risk: changing the
computation while extracting configs and changing paths. They support delivery
of runnable, traceable code. They do not establish the scientific validity of
the paper or justify additional experiments.

## Default-setting checks

For all seven models, source constants, default configuration values, baseline
estimator parameters, both label maps, and the full 50-candidate sequence under
each original outer-fold seed matched. The KNN sequence check used a training
size sufficiently large not to exclude any published neighbor count.

## Executed reduced-workload checks

Dataset: deterministic synthetic data, 20 groups, 240 samples, eight numeric
nonconstant features plus one constant feature, with original labels 1 and 2
both present within every group. This is not the research dataset and is not
included among the released scientific materials.

Both original and new versions executed all five outer folds, three candidate
trials per fold, and final-model fitting. Numerical threads and supported
estimator jobs were limited equally to 1. XGBoost was limited to 60 trees and
8 early-stopping rounds only for this test. All published configs remain at
the source defaults.

All numeric fields in compared CSVs and JSONs matched with absolute and relative
tolerance `1e-12`. Runtime dates, elapsed times, sanitized paths, and the clarified
XGBoost model-description text were excluded from equality checks. For serialized
model bundles, feature lists, label maps, best parameters, class predictions,
and probabilities/decision scores were compared. Original result filenames and
columns remained intact; only three run-metadata JSON files were added.

| Model | Result | CSVs compared | JSONs compared | Model bundles compared |
|---|---|---:|---:|---:|
| xgboost | Passed | 11 | 7 | 1 |
| logistic_regression | Passed | 11 | 7 | 6 |
| gaussian_nb | Passed | 11 | 7 | 6 |
| knn | Passed | 11 | 7 | 6 |
| linear_svm | Passed | 11 | 7 | 6 |
| decision_tree | Passed | 11 | 7 | 6 |
| random_forest | Passed | 11 | 7 | 6 |

CLI help and default dry-run succeeded from outside the repository directory.
Repeated use of a populated output directory was rejected without modifying the
existing summary. Shareable run JSON files did not contain the external input
workspace's absolute directory. Restored prediction labels stayed in 1/2.

## Sensitive-information inspection

Pattern-based inspection covered 22 files: the seven supplied scripts,
eight release Python files, and seven release JSON configurations. No matches
were found for the inspected private-key, token, credential-assignment,
personal-path, email, IP-address, or internal-node patterns. This is a bounded
inspection, not a guarantee against all confidential content. Server files,
real data, logs, hidden local files, and Git history were not provided or audited.

## Not verified

No real Task 2 dataset or original paper environment was supplied. Full
50-trial real-data training, paper AUC values, optional LR SHAP, legacy dependency
fallbacks, pixel equality of output plots, upstream standardization, scientific
label definitions, and whole-study leakage status were not verified. Paths and
console output remain usable for local execution; tracebacks/logs are not covered
by the shareable-JSON redaction guarantee.

The JSON companion records source/release hashes, exact candidate sequence
hashes, observed versions, scope, and per-model status. It does not include
research data, raw original code, or synthetic AUC claims for publication.
