# Change log: Task 1 evaluation release

## Scope and preservation

This release starts from the 15 supplied evaluation/preparation/ablation source
files listed with SHA-256 hashes in `file_mapping.json`. It is an additive
release, not a replacement for earlier training packages.

The seven bootstrap statistical and validation function/class ASTs, excluding
`parse_args` and `main`, were compared with their respective originals and were
unchanged. Source-specific threshold rules, CV consistency behavior, sampling
order, percentile calculations, output CSV names and contents are preserved.
No sampling or ML method has been silently standardized across the scripts.

## Interface changes

1. Added JSON-backed argument parsing with repository-relative configuration
   paths and explicit working-directory-relative CLI paths.
2. Added a shared `_eval_runtime.py`, with no numerical evaluation kernel inside.
3. Bootstrap requires a specific `--result-dir` unless `input.result_dir` is set
   in its JSON. It no longer assumes that a script was copied into a result
   directory. No latest-run heuristic was added.
4. Added explicit `--output-dir`, `--output-file`, `--dry-run`, and ablation
   `--data`, `--n-jobs`, `--n-trials`, `--seed` interfaces as applicable.
5. Existing populated result directories/prepared CSVs are refused instead of
   being overwritten. Original research inputs are not modified.
6. Added successful-run config/environment/reference records. Original bootstrap
   metadata now redact external absolute directory prefixes. Raw console logs
   and tracebacks are outside that redaction scope.
7. Moved preparation column lists/regexes and ablation parameters to four paired
   JSON configurations. The default numerical values and candidate-key order
   remain equal to the corresponding uploaded sources.
8. Changed the inner-split optional `max_retries` default from an import-time
   constant to `None` resolved from the active configuration. Its original default
   value is retained, and CLI/config overrides are no longer silently shadowed.
9. Removed global suppression of warnings from the ablation initialization block;
   warning visibility changes, not numerical operations.
10. Replaced obsolete developer-era usage/header wording. Clarified that individual
    outer-fold prediction CSVs are saved even though no merged OOF/SHAP analysis
    is performed. Model/console description strings reflect that distinction.

## Explicit filename corrections

| Source preparation | Old output | Old fitting input / new aligned basename |
| --- | --- | --- |
| `remove_W_0_1.py` | `removeW_1_2_feature_labels_standard.csv` | `removeW.csv` |
| `removeInter_0_1.py` | `removeInter_1_2_feature_labels_standard.csv` | `remove_Inter.csv` |

The corrected files now live under `data/task1/ablation/`. This is an I/O
correction only; no labels, feature values or deletion rules are changed.

## Deliberately unchanged or absent

- `0/1` input label identity mapping.
- Existing scaling assumptions; no extra scaler, imputation or feature selector.
- Distinct `>= 0.5`, `> 0.5` and `> 0` prediction contracts.
- XGBoost warning-based vs. the other classifiers' strict bootstrap validation.
- Original per-ablation hyperparameter retuning, fitting subsets and full-data
  final model behavior.
- The display-only historical `BASELINE_AUC = 0.7931`; it was not independently
  validated. JSON `null` disables this comparison without changing model fitting.
- No new bootstrap forest-plot script; the supplied CI scripts generate tables.
- No full-feature XGBoost retraining source is duplicated into this add-on.
- **No XGBoost SHAP implementation was supplied or invented.** Only a placeholder
  README documents the missing source.
- No automatic secret removal from Git history or arbitrary result/log files.

The scoped pattern review found no matches requiring credential removal in the
supplied/released code or configurations. This is a limited file review, not a
security guarantee for the user's broader workspace.
