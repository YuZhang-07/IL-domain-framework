# task1: model preparation for SHAP

`retrain_xgboost_for_shap.py` refits the final XGBoost model using the supplied
protocol. Raw input labels: **0/1**. The source-specific label mapping is
fixed in this entry point and is not changed by the JSON configuration.

This is the **model preparation** stage only: it does not import a SHAP
explainer, compute SHAP values, or generate SHAP plots. The attribution/plotting
source used in the manuscript has not been supplied with this refit release.
`features_used_by_model` records tree split usage, not SHAP attribution.

From the repository root:

```bash
python scripts/evaluation/task1/shap/retrain_xgboost_for_shap.py \
  --cv-results results/task1/xgboost/run_01/cv_xgb_results.csv
```

Replace `run_01` with the actual training run. There is no automatic selection
of the latest run. `--data` overrides the task-specific default input;
`--output-dir` (`--outdir`) overrides the timestamped output directory.

Configuration: `configs/task1/evaluation/shap/refit_xgboost.json`.

See [English instructions](../../../../docs/shap_model_refit/README.md) or
[中文说明](../../../../docs/shap_model_refit/README_zh-CN.md).
