# Changes from the supplied training scripts

## File mapping

| Supplied source | Public entry point | Executed configuration |
|---|---|---|
| `fin_01_xgboost_opt(1).py` | `scripts/training/task1/train_xgboost.py` | `configs/task1/xgboost.json` |
| `fin_0_1_logistic_regression(1).py` | `scripts/training/task1/train_logistic_regression.py` | `configs/task1/logistic_regression.json` |
| `fin_0_1_NB(1).py` | `scripts/training/task1/train_gaussian_nb.py` | `configs/task1/gaussian_nb.json` |
| `fin_0_1_knn(1).py` | `scripts/training/task1/train_knn.py` | `configs/task1/knn.json` |
| `fin_0_1_svm(1).py` | `scripts/training/task1/train_linear_svm.py` | `configs/task1/linear_svm.json` |
| `fin_0_1_decision_tree(1).py` | `scripts/training/task1/train_decision_tree.py` | `configs/task1/decision_tree.json` |
| `fin_0_1_random_forest(1).py` | `scripts/training/task1/train_random_forest.py` | `configs/task1/random_forest.json` |

## Publication-interface changes

1. Moved constants, search spaces, baseline parameters, and fixed estimator
   settings into the matching JSON. Dictionary key order and exact extracted
   numeric values are retained because the original samplers can depend on order.
2. Added the shared `_runtime.py` and the same CLI options to every model. The
   original `main()` training orchestration is now called `run_training()` and is
   invoked by a configuration-aware `main(argv=None)`.
3. Replaced shell-working-directory-dependent default input lookup and
   script-directory output placement with repository-relative defaults. Explicit
   CLI input/output paths are supported.
4. Added unique timestamped result directories and refusal to reuse a populated
   explicit output directory. No overwrite/delete command was added.
5. Added shareable run settings, version records, and input/code/configuration
   hashes. The input-path field in the original summary now uses a relative or
   redacted reference. No environment-variable dump, user name, or host name is
   collected by the new runtime module.
6. Moved random seeding from module import to execution after settings are read.
   Retained the same default seed. Removed the original blanket warning
   suppression so convergence/deprecation warnings remain visible.
7. Changed the inner-split `max_retries` default to resolve from the current
   setting at call time. Its default value remains 50; this prevents a changed
   JSON setting from being ignored by Python's definition-time default binding.
8. Added configurable job counts where already supported by estimators and a
   positive-value numerical-thread cap. The default remains -1.
9. Deferred the optional LR SHAP import until SHAP is actually requested. The
   original `RUN_SHAP=False`, explanation procedure, and output logic remain.
10. Replaced obsolete development-history introductions and unrelated old AUC
    references. Corrected the XGBoost description: per-fold held-out prediction
    files exist, although pooled OOF statistics/SHAP are not computed. Updated
    fold-count text in summaries to reflect the configured value.

The preexisting plotting functions remain inside the individual training
scripts. They were not split into a new plotting subsystem. The seven
model-specific training implementations were not replaced with a new generic
training framework.

## Unchanged numerical/scientific decisions

Existing 0/1 label mapping; input values and row order; positional feature
selection; per-model missing-value handling; outer GroupKFold; one inner holdout
with the original fallback/retry logic; outer-training near-constant filtering;
parameter candidates/order/selection/tie behavior; XGBoost early stopping and
class weighting; inner-training-only fitted CV estimators; metric and class-label
rules; model-specific full-data refitting; original prediction column names,
result filenames, and plotting logic.

No new scaler, imputer, class merger, feature selector, optimizer, ensemble,
threshold search, or biological class definition has been introduced.

## Bounded sensitive-content inspection

All seven supplied sources were inspected for personal absolute paths, explicit
credential assignments, recognized GitHub token formats, private-key blocks,
email addresses, and numeric IP addresses. No matches were found in those sources.
Therefore this release does not claim that preexisting passwords or private keys
were deleted. The actual path work concerns portability and preventing local
absolute input directories from being written into the new shareable JSON files.

The inspection did not include a live GitHub repository, Git history, real input
CSVs, model binaries, or external logs. The absence of inspected patterns is not
a formal proof that all sensitive information is absent. No unedited original
source, test log, synthetic test output, or model binary is in the release.
