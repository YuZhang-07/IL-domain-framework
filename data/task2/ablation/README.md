# Task 2 ablation inputs

The paired preparation scripts read `data/task2/proid_1_2_feature_labels_standard.csv`
and write `removeW.csv`, `remove_Inter.csv`, `remove_resn.csv`, or `remove_hbond.csv`
here. Labels remain original 1/2. The fitting scripts expect an already-ablated
CSV; they do not delete columns themselves. Existing ablated CSVs can be supplied
with `--data` without regenerating them.

Generated tables are ignored by this local `.gitignore`. This is not a statement
that analysis data should never be shared: publish the manuscript dataset through
the agreed repository/archive and document its access separately.
