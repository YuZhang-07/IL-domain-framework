# Verification scope

The files in this release were compared with the seven supplied training scripts
in the same local environment. The comparison is **not** a reproduction of the
manuscript's numerical results.

## Checks completed

All eight Python files parse successfully. All seven entry points accept
`--help` and `--dry-run` when launched from a different working directory.
The dry runs create no result directories. An invalid job count and a
configuration for the wrong model are rejected. A populated output directory is
rejected without altering its existing contents. The LR public script also
completed training in a fresh subprocess without importing an original script.

For every model, the extracted default scalar settings, baseline settings,
search-space values, default estimator parameters, and the ordered 50-candidate
sequence for all five default outer-fold seeds match the supplied implementation.
The inner-holdout indices also match for all five outer folds on the synthetic
input.

## Reduced-workload numerical comparison

The synthetic input has 240 rows, 20 protein-like groups, and 7 numerical features
(including 1 constant feature). Every group contains both labels 0 and 1.
The fixture-generation seed is 1729. It is not research data.

Both versions of each script used five outer folds, three candidates per outer
fold, seed 42, and one numerical thread. XGBoost was temporarily limited to 12
trees and 3 early-stopping rounds in both versions. These testing settings were
not written into the shipped experiment configurations, which retain the original
50 candidates / 3000-tree cap / 100-round early stopping defaults.

| Model | Fold metrics and selected parameters | All fold prediction tables | Final-model predictions |
|---|---|---|---|
| XGBoost | Match | Match | Match |
| Logistic regression | Match | Match | Match |
| Gaussian NB | Match | Match | Match |
| KNN | Match | Match | Match |
| Linear SVM | Match | Match | Match |
| Decision tree | Match | Match | Match |
| Random forest | Match | Match | Match |

Numerical comparisons use relative and absolute tolerances of 1e-12. Column names,
row order, selected-feature names, and parameter selections were also checked.
The new JSON records completed status and the correct input SHA-256; the absolute
external test input directory is absent from the shareable JSON files.

## Environment and limits

The local comparison used Python 3.13.5. Package versions are recorded in
`verification_results.json` and `requirements-tested.txt`. These are observed
testing versions, not the unprovided historical manuscript environment.

The real feature CSV, historical result tables, all-50-candidate full-workload
training, optional LR SHAP branch, cross-version/cross-platform equivalence,
existing Git history, and external datasets/logs were not verified. No claim is
made that the paper AUC values have been reproduced.

The bounded sensitive-content pattern scan found no matches in the eight public
Python files and seven configuration JSON files. It does not establish the
absence of all conceivable sensitive information.

Machine-readable records: [verification_results.json](verification_results.json).
