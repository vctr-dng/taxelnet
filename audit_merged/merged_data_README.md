# `data/newton/v1/merged/` (indenter + grasp training pair)

Written by `scripts/merge_training_data.py` (branch `worktree-grasp-merge`) from
`data/newton/v1/training/`; see `merge_manifest.json` for source/output checksums.

| File | Rows | Columns |
|---|---|---|
| `Dataset_Merged_Real.csv` | 17,372 | `force, Path, dataset, tax1 ... tax28` |
| `Dataset_Merged_Simulation.csv` | 17,372 | `force, Path, dataset, n195 ... n14899` |

- Rows 0-12,729: `Dataset_Indenter_*` copied verbatim (`dataset` 01-06).
- Rows 12,730-17,371: `Dataset_Grasp_*_qtac100` (2,321 grasps x pads S1/S2, 14 objects). Each `Name` becomes
  `Path = <Shape>_<Size>/<rest>`, e.g. `Cuboid_G/20_02_A_V+1_H+1_Deg15_S1`, with `dataset = grasp`.
  The real `std1 ... std28` columns (temporal std over the end of the hold) are dropped.
- Grasp `force` is the gripper setting label: 20 = minimum (about 20 N measured) and 235 = maximum
  (about 90 N measured, per the dataset author). The grasp data come from the gripper pads, not the
  indenter sensor, and pads S1 and S2 respond differently (see the audit).
- No key, object or row overlaps with the test set `data/newton/v1/test/` (its objects are `*_Test*`);
  checked 2026-10-15.
- With `--data.split_strategy shape`, a grasp row's "shape" is its object (`Cuboid_G`, ...), so both pads
  of a grasp stay in the same partition. With `group`, S1 and S2 of one grasp may be split apart.
- Use: `--data.data_dir data/newton/v1/merged --data.real_csv Dataset_Merged_Real.csv
  --data.sim_csv Dataset_Merged_Simulation.csv --data.cache_dir data/newton/v1/merged/.cache`.
