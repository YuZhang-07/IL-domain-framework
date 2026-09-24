# Task 2 publication changes

The model functions originate from the seven Task 2 sources supplied for this
request. The CLI/config helper follows the prior Task 1 release's interface.

## Publication interface

- Rename entrypoints to descriptive `train_*.py` names and replace development
  history in module headers with input, label, and execution documentation.
- Move source constants/search spaces/fixed estimator arguments to matching
  JSON files. JSON key and candidate-list order are preserved.
- Add explicit input/config/output/thread/seed/trial CLI arguments and a dry run.
- Resolve default paths from the repository layout; explicit CLI relative paths
  use the shell working directory.
- Create timestamped result directories and refuse populated explicit outputs.
- Record sanitized run settings, observed versions, SHA-256 digests, and status.
- Replace the inner split function's import-time retry default by a call-time
  default so JSON overrides really take effect. Default value remains unchanged.
- Remove blanket suppression of warnings. Load optional LR SHAP lazily only when
  requested. Do not replace the original SHAP calculation with a different method.
- Retain XGBoost's historical numeric comparison reference; allow explicit null
  to disable reporting. Clarify its status; do not represent it as a new result.
- Clarify XGBoost's existing per-fold predictions versus absent pooled OOF metric.
- Add task-local data ignore rules and task-local docs rather than overwrite root
  files or any Task 1 files.

## Numerical behavior left unchanged at defaults

Original 1/2 to 0/1 mapping, positive class direction, CSV column positioning,
feature values, source missing-value policies, outer and inner splits, random
seeds, candidate sampling order, near-constant filtering, validation-only
selection, fitting sample sets, early stopping, decision thresholds, metric
formulae, output columns, and final-model logic are retained.

The source LR grid still requires 50 total grid combinations before slicing
by N_TRIALS. This source-specific behavior is documented rather than silently
replaced by a general search engine. Changing the grid itself requires taking
that constraint into account. XGBoost's legacy-version fallback and other
source edge-case behaviors have not been redesigned in this packaging task.

No full-training script was moved into configs, no new classifier was added,
and no existing scientific result was fabricated or recomputed from absent data.
