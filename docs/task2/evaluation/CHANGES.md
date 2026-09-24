# Change log: Task 2 evaluation release

## Source scope

Seven bootstrap programs, four feature-removal programs and four XGBoost ablation
programs were supplied. Source and release hashes are in `file_mapping.json`.
Task 2 fitting and bootstrap programs were edited from these actual sources,
not obtained by blindly relabeling Task 1 code. The existing release layout and
path/configuration helper were reused with an independent Task 2 namespace.

## Interface changes

1. Added JSON-backed CLI configuration. JSON-relative paths use the repository
   root; explicit CLI-relative paths use the working directory.
2. Moved code to `scripts/evaluation/task2/`, configurations to
   `configs/task2/evaluation/`, and documents to `docs/task2/evaluation/`.
   No earlier training or Task 1 file is replaced.
3. Bootstrap requires a specific result directory; no copy-into-results or
   latest-run heuristic is used. Added an explicit output-directory override.
4. Added new/empty output guards and exclusive creation for preparation CSVs.
5. Added successful-run configuration/environment/reference records. Redacted
   external directory prefixes in shareable JSON output; console/traceback
   redaction is outside the scope of this helper.
6. Extracted original scalar feature lists, regexes, numerical defaults, ordered
   search spaces and baseline/fixed parameters into paired JSON configurations.
7. Added preparation `--data`/`--output-file`; fitting `--data`/`--output-dir`,
   `--n-jobs`, `--n-trials`, `--seed`; all entrypoints support `--dry-run`.
8. Resolved the optional inner-split `max_retries` from the active configuration
   at call time instead of capturing the import-time default. Default is still 50.
9. Removed the ablation initialization block's blanket warning suppression.
   This changes warning visibility, not the numerical operations.
10. Replaced developer-era header/usage descriptions and clarified that individual
    outer-fold predictions are saved even though no merged OOF or SHAP analysis
    is performed. Preserved original output filenames/columns. Bootstrap's
    documentation-only `source_training_script` points to the released entrypoint.
11. Preserved the original display-only historical AUC reference, while allowing
    JSON `null` to disable that comparison. This does not alter fitting/selection.

## Explicit preparation filename alignment

| Original preparation | Original output | Original fitting input / new basename |
| --- | --- | --- |
| `remove_W(2).py` | `removeW_1_2_feature_labels_standard.csv` | `removeW.csv` |
| `removeInter(1).py` | `removeInter_1_2_feature_labels_standard.csv` | `remove_Inter.csv` |

Files now live under `data/task2/ablation/`. This is an I/O change, not a label,
row-order, feature-value, or deletion-rule transformation. Existing old-named
inputs can still be specified via the fitting `--data` option.

## Deliberately unchanged or absent

- Original labels 1/2 and explicit internal mapping {1: 0, 2: 1}; original label 2
  is positive. Preparation never changes the label values.
- Source-specific score columns, threshold ties and strict/warning-based input
  consistency behavior. Six strict scripts and the XGBoost warning policy were
  not replaced by a single generic numerical implementation.
- Task 2 KNN tolerance `1e-8` and LR tolerance `1e-7` are taken from these sources.
- Within-fold cluster resampling, multiplicity computation, fold averaging,
  percentile intervals, redraw order and default 2000 replicates.
- Existing standardized-input assumption; no scaler, imputer or other selector
  was added to the scientific workflow.
- Original per-ablation retuning and final full-data model behavior.
- Historical `BASELINE_AUC=0.7931` was not independently validated and should not
  be assumed to be the manuscript's final full-feature baseline.
- **No XGBoost SHAP computation script was supplied or invented.** Only a missing-
  source README is present. This release cannot reproduce an unspecified SHAP
  analysis.
- No new forest-plot workflow, research data, fitted models, original-script
  duplicates, broad audits or Git-history rewriting are included.

See `VERIFICATION.md` for the actual bounded source-to-release tests and their
limits. Scientific validation of the manuscript data is not claimed.
