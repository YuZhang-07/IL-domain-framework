# Task 1 ablation tables

Generated locally by the paired preparation scripts:

- `removeW.csv`: remove the eight named whole-protein scalar features.
- `remove_Inter.csv`: remove the five named interfacial scalar features.
- `remove_resn.csv`: remove columns matching `^[A-Z]{3}-[A-Z]{3}$`.
- `remove_hbond.csv`: remove columns matching `^hbond[A-Z]{3}-[A-Z]{3}$`.

No table is bundled with this release. Default full input:
`data/task1/proid_0_1_feature_labels_standard.csv`.
The script/configuration paths are documented in `docs/task1/evaluation/README.md`.
Ignoring generated files is a local Git workflow choice, not a publication
access restriction or a guarantee against a manual upload.
