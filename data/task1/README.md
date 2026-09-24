# Task 1 input data

Place `proid_0_1_feature_labels_standard.csv` here, or pass an input file using
`--data`. No research data are bundled in this training-code package.

Inherited column contract: first column = protein grouping ID; last column =
existing binary label 0/1; all intermediate columns = numeric model features.
The scripts do not add a scaler, merge classes, or remove metadata columns.

The input-file hash is recorded during a run so the exact input and row ordering
can be identified. Supply the actual data-archive citation here when the research
data release exists; no archive URL, DOI, or availability claim is invented.
