# Task 2 input data

The training CSV is not distributed in this code-only add-on.

Default local path (relative to the repository root):

```text
data/task2/proid_1_2_feature_labels_standard.csv
```

Alternatively, provide a different local filename using `--data`.
The seven scripts retain the positional input schema of the supplied sources:

| Position | Meaning |
|---|---|
| First column | Protein/group identifier; normally named `protein_id` |
| Middle columns | Existing standardized numeric features, in their original order |
| Last column | Original integer label `1` or `2` |

Internal mapping is `{1: 0, 2: 1}`. `true_label` and `pred_label` are exported
in original `1/2` notation. Positive probabilities and SVM decision scores
refer to original class `2`.

No scaler or imputer is added. Do not insert a `frame_id`, CSV index, or other
metadata column among the features: the original loader treats every middle
column as a model feature. Keep row order unchanged to reproduce the same
source workflow. Different frames within a protein may have different labels;
the scripts preserve those row labels and group by the first column.

The local `.gitignore` excludes data files from newly tracked files in this
folder. It is not an access-control or redaction system. Deliberately published
data should have an explicit release record, schema, version, and access link.
