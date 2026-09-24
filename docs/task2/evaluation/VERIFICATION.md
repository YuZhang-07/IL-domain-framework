# Task 2 evaluation release: verification scope

## What this report does and does not establish

These are executed, bounded source-to-release checks for the 15 supplied Task 2
scripts. The reference is each uploaded source, not Task 1 or a generic rewrite.
The machine-readable results are in `verification_results.json`.

**No real Task 2 feature matrix or prediction CSV was supplied for this release.**
No manuscript AUC, confidence interval, ablation magnitude or SHAP conclusion was
recomputed. Small synthetic equivalence checks are not proof of correctness for
all inputs, software versions, hardware, or the original scientific design.

No actual XGBoost SHAP computation was found among these 15 inputs. SHAP was not
implemented, substituted or executed.

## 1. Static preservation and executable entrypoints

All 16 released Python files compiled and all 11 configuration JSONs parsed.
All 15 executable entrypoints passed `--help` outside the repository working
directory. Source and release hashes matched `file_mapping.json`.

For each of seven bootstrap programs, 15 original function/class ASTs were
unchanged; only `parse_args` and `main` were excluded. Thus the numerical kernels,
label mapping, validation policies, multiplicity ordering and CI calculations
were not silently merged or rewritten.

For each ablation fitting program, 12 of 16 original function ASTs were unchanged.
The four documented interface adaptations are estimator parameter references,
active-config retry default resolution, shareable summary paths/display strings,
and main/CLI wrapping. Original scientific defaults and estimator parameters
were also checked by loading both versions, and all default 50-candidate sequences
for each of the five outer-fold seeds matched.

## 2. Bootstrap numerical comparison

Both versions of every classifier were run on a five-fold synthetic prediction
fixture with 20 protein clusters and 240 frames. All clusters contained original
labels 1 and 2, with differing label proportions. Threshold ties were included
(0.5 for probabilities, 0 for SVM decision scores). Matching CV result tables
were derived from these fixtures.

Each version executed the default **2000 resamples**, seed 42 and confidence 0.95.
Compared the original four CSV outputs, including fold/CV comparison columns
where applicable, composition tables, all replicate statistics and CI summaries.
Comparison used `rtol=atol=1e-12`.

| Classifier | Result | Maximum absolute CSV difference |
| --- | --- | --- |
| xgboost | Passed | 0.0 |
| decision_tree | Passed | 0.0 |
| gaussian_nb | Passed | 0.0 |
| knn | Passed | 0.0 |
| linear_svm | Passed | 0.0 |
| logistic_regression | Passed | 0.0 |
| random_forest | Passed | 0.0 |

This tests preservation of fixed-prediction analysis, not model retraining or
uncertainty estimates on manuscript data.

## 3. Feature removal comparison

Both original and release preparation programs were executed on a synthetic
240-row table containing 433 source-named features plus group and label columns.
The table included the eight global and five interface scalar names, all 210
unordered contact-pair names, and their 210 hydrogen-bond counterparts.

Row order, remaining column order and values, labels and group identifiers were
compared. Labels were never remapped by preparation.

| Variant | Removed features | Remaining features | Maximum absolute CSV difference |
| --- | --- | --- | --- |
| remove_global | 8 | 425 | 0.0 |
| remove_interface | 5 | 428 | 0.0 |
| remove_contacts | 210 | 223 | 0.0 |
| remove_hbonds | 210 | 223 | 0.0 |

The global/interface output basenames were deliberately aligned with the original
fitting inputs. Numerical comparisons account for this documented filename change.

## 4. Ablation fitting comparison

All four variants ran all five outer folds and the final full-data fitting stage
for both their original and released programs. Test-only settings: **3 candidates
per fold, 60-tree cap, 8-round early stopping, 1 thread**, with the original seed,
group-aware split logic, search order and all other model parameters preserved.
The original reference was controlled only in memory by a test launcher; its
uploaded file was not overwritten.

Compared five outer-fold prediction tables, per-fold trial tables, CV metrics,
selected-parameter JSONs, removed-feature lists and the final model's probabilities.
Comparison used `rtol=atol=1e-12`.

| Variant | Result | Maximum absolute CSV difference | Final probability maximum difference |
| --- | --- | --- | --- |
| remove_global | Passed | 0.0 | 0.0 |
| remove_interface | Passed | 0.0 | 0.0 |
| remove_contacts | Passed | 0.0 | 0.0 |
| remove_hbonds | Passed | 0.0 | 0.0 |

**Published configurations retain 50 candidates, 3000-tree cap and 100-round
early stopping.** Full-size fitting under these defaults was not run here;
default configuration/estimator equality and full candidate generation order
were checked separately.

## 5. Packaging and privacy checks

Existing populated bootstrap and fitting output directories were refused.
Existing prepared CSVs were refused. CLI-relative path resolution, repository-
relative defaults and no-write dry runs passed. Generated shareable JSONs did
not contain the external test directory prefix. The release member paths did
not collide with the earlier Task 1 training, Task 2 training or Task 1 evaluation
ZIP member paths.

A limited pattern review covered **15 original scripts, 16 released Python files
and 11 JSON configurations**. It found no matches for the reviewed private-key
blocks, common credential/token assignments, cloud/GitHub token patterns,
personal absolute paths, private IPv4 patterns, explicit internal-node patterns
or email addresses. No matching credential value needed to be removed.

This is not a security guarantee. Console output/tracebacks are not automatically
sanitized. Basenames, feature names and scientific contents are intentionally
retained. Other files, research-data contents and Git history were not reviewed.

## 6. Observed environment

Python: **3.13.5**. Package versions are recorded in
`requirements-tested.txt` and `verification_results.json`. They are the actual
packaging-test environment, not the unavailable historical manuscript environment.

## 7. Remaining scope limits

- No real-data performance or CI values were checked.
- No full-size default 50-candidate/3000-tree run was performed.
- No old-version fallback, GPU, Slurm or multi-node execution was tested.
- No actual XGBoost SHAP workflow was supplied or tested.
- The historical display reference AUC 0.7931 was preserved, not independently
  verified as the manuscript's final full-feature baseline.
- Original warning/strict validation distinctions and input assumptions were
  retained; a packaging test does not certify every aspect of the source method.
