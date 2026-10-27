# Per-sensor target normalization (grasp-only, cnn_v0)

Branch `lab/sensornorm`. Training data: grasp training CSVs only (4,642 rows, S1/S2 2,321 each), built by
`scripts/build_sensor_data.py` (Path = `<Shape>_<Size>/<rest>`, `sensor` column in the real file).
Shape split (seed 0), 300 epochs, checkpoint chosen on val macro MAE (held-out grasp objects), seeds 0-2.
MLflow experiment `taxelnet-sensornorm`; launcher `run_arms.sh`, table `summarize.py`, raw `results.json`.

## Implementation
- `PreprocessingConfig.target_group_key` (`none` | `sensor`). With `sensor`, `target_scaler` (`global_scale` or
  `per_taxel_zscore`) is fitted separately on each pad's training rows; `Preprocessor.transform_y/inverse_y` take
  per-row labels (`group=`), parameters are serialized with a leading group axis plus `group_labels`.
  Labels come from the `_S1`/`_S2` suffix of Path/Name (`PairedData.sensor`). Unchanged behaviour with `none`
  (tests added). The network input is the deformation grid only; sensor id is not an input.
- Semantics: the net predicts in a sensor-normalized space, so the sensor gain is absorbed by the scaler and not
  learned. This is legitimate at deployment only because the pad id is known.

## Arms
A control (global_scale); B per-sensor global_scale; C per-sensor per_taxel_zscore; D = B with the 16 rows having
any taxel > 10,000 TC dropped (train+val only; test untouched). Mean ± std over 3 seeds, TC.

| arm | n | best ep | val MAE | val S1 | val S2 | test MAE | 4-grp mean | calib k=5 | test S1 | test S2 | Cuboid | Sphere | Cyl M | Cyl Buttons (S2 scaler) | Cyl Buttons (S1 scaler) | cos | aligned | peak hit | SSIM | amp logratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A control: global_scale | 3 | 108.7 ± 65.0 | 230.5 ± 5.5 | 214.8 ± 6.6 | 246.1 ± 16.2 | 130.9 ± 2.5 | 150.0 ± 3.9 | 133.8 ± 0.8 | 127.3 ± 1.1 | 151.9 ± 13.5 | 201.1 ± 7.4 | 73.1 ± 5.0 | 173.8 ± 4.4 | 151.9 ± 13.5 | – | 0.854 ± 0.008 | 110.1 ± 1.8 | 0.417 ± 0.031 | 0.639 ± 0.009 | -0.106 ± 0.024 |
| B per-sensor global_scale | 3 | 144.7 ± 38.1 | 231.6 ± 3.9 | 223.1 ± 3.4 | 240.1 ± 7.9 | 132.9 ± 1.7 | 157.8 ± 3.8 | 127.1 ± 2.2 | 124.6 ± 1.6 | 182.0 ± 17.2 | 201.1 ± 3.2 | 65.1 ± 1.3 | 182.9 ± 5.2 | 182.0 ± 17.2 | 96.1 ± 4.2 | 0.872 ± 0.007 | 105.1 ± 1.5 | 0.456 ± 0.014 | 0.562 ± 0.017 | -0.548 ± 0.031 |
| C per-sensor per_taxel_zscore | 3 | 77.0 ± 25.6 | 262.2 ± 0.3 | 252.2 ± 0.8 | 272.3 ± 1.5 | 166.8 ± 2.6 | 199.1 ± 4.9 | 149.4 ± 0.8 | 146.8 ± 0.5 | 284.6 ± 19.8 | 226.3 ± 1.6 | 88.3 ± 1.0 | 197.2 ± 1.6 | 284.6 ± 19.8 | 103.8 ± 4.4 | 0.834 ± 0.013 | 128.2 ± 2.6 | 0.398 ± 0.032 | 0.423 ± 0.008 | -0.771 ± 0.040 |
| D = B, 16 glitch rows (>10k TC) dropped | 3 | 131.0 ± 28.7 | 244.2 ± 0.5 | 242.7 ± 2.9 | 245.7 ± 3.6 | 145.5 ± 0.7 | 174.9 ± 1.9 | 127.7 ± 1.3 | 131.4 ± 1.7 | 228.1 ± 14.0 | 216.0 ± 3.0 | 68.7 ± 0.7 | 186.8 ± 2.9 | 228.1 ± 14.0 | 90.1 ± 2.2 | 0.870 ± 0.001 | 105.9 ± 0.2 | 0.458 ± 0.009 | 0.473 ± 0.011 | -0.806 ± 0.020 |
| zero predictor | – | – | – | – | – | 176.0 | – | – | 186.8 | 112.3 | 293.8 | 105.9 | 262.0 | 112.3 | – | – | – | – | – | – |

Notes: "test MAE" = macro over test groups (all 358 rows); "4-grp mean" = mean of the four object-group MAEs;
calib k=5 = global-gain calibration, object-macro. Zero predictor: 176.0 overall, 186.8 on S1 rows, 112.3 on
Cylinder_TestButtons. Reference (cnn_v0 on merged indenter+grasp, different training data): test 151, calib k=5
137, Cuboid 187, Sphere 98, Cyl M 180, Buttons 229.

## Findings
- Per-sensor global_scale (B) does not improve overall test MAE (132.9 vs 130.9, within noise) but helps pattern
  metrics (cosine 0.872 vs 0.854, aligned MAE 105 vs 110, peak hit 0.46 vs 0.42), S1 rows slightly (124.6 vs
  127.3) and Sphere (65 vs 73). Calibrated k=5 improves: 127.1 vs 133.8.
- Per-taxel per-sensor z-score (C) is clearly worse everywhere (test 166.8); noisy per-taxel stds hurt.
- Cylinder_TestButtons (the only S2 test object) is worse with the S2 scaler (B 182, C 285, D 228, vs control
  152; zero predictor 112). Coordinator's finding: its signal looks S1-like (sum ~2,249 TC). Sensitivity, scoring
  it with the S1 scaler at inference (no retraining): B 96.1 ± 4.2, C 103.8 ± 4.4, D 90.1 ± 2.2, all below the
  zero predictor (112) and the control (152). So the S2 label hurts and S1-scaler scoring helps, consistent with
  that finding. With the S1 scaler B's 4-group mean would be about 111 (Buttons 96 instead of 182).
- Val S2 error is high in every arm (240-272) and per-sensor normalization does not reduce it (B 223/240 for
  S1/S2 vs A 215/246).
- D (dropping 16 glitch rows) lowers the S2 std but did not help test MAE (145.5) or val (244): fewer training
  rows/objects in a small set and a different split; val numbers are on slightly different rows, so not strictly
  comparable to A/B.
- Best epochs vary widely (A: 20-174, B: 105-196), so seed variance in checkpoint selection is large; the
  val curve is flat and noisy.
- Compared with the merged-data reference (151 test, Cuboid 187, Sphere 98), grasp-only training is better
  overall (131 control) and on Sphere (73), but Cuboid is worse (201). Not strictly comparable (different data).

## Caveats
- Cylinder_TestButtons is the single S2 test object, so every S2 conclusion rests on one object, and its
  S2 label seems wrong or atypical (above). Test-time sensor assignment matters more than the arm.
- 306 glitch-like rows (taxel > 3,000 TC, mostly S2) remain in A/B/C; D removes only the 16 above 10k TC.
- Overall deltas A vs B are within seed noise (std 1.7-2.5); only 3 seeds and one split.
- Follow-up arm not run: feed the sensor id (or the S1/S2 gain) to the network.
