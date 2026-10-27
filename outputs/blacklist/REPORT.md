# Blacklist of high grasp readings, with per-test-pad calibration (grasp-only)

Branch `lab/blacklist` (from `lab/capacity`), 2026-10-22. MLflow experiment `taxelnet-blacklist`.
Files: `make_lists.py` (blacklists), `run_arms.sh` (launcher), `scripts/pad_calibrate_eval.py` (calibration +
summary), `results.md` / `results.json` (both arms), `reference_capacity/` (earlier best model, same protocol).

## Setup
- **Data:** grasp training CSVs only (`scripts/build_sensor_data.py --exclude-names`).
  - **glitch:** drops 26 rows, 4,616 kept. A glitch is a row with a peak above 10,000 TC, or a peak more than 3× its largest 4-neighbour with at most 2 taxels above 3,000 TC: 4 S1 + 22 S2 rows (`flagged_classified.csv`).
  - **all160:** drops every row of `flagged_grasp_high_readings.csv`, 4,482 kept. The other 134 rows have a peak of 3,005–9,715 TC (median 4,654) over several taxels, mostly S2 at grip 235.
- **Model:** footprint_cnn + CoordConv residual (previous best). Target scale fitted per training pad (`target_group_key sensor`).
- **Split:** shape split, seed 0. Val = Cuboid_G (308 rows, grip 20 only, no flagged row), identical in both arms. 300 epochs, best-val checkpoint, seeds 0–4.
- **Test calibration:** the test pads are not the training pads. Each test pad (S1: Cuboid, Sphere, Cyl M; S2: Buttons) gets one L1 gain (`global_gain`) fitted on k of its own readings, scored on all other test readings, 100 draws.
  - `s1_xobj`: the S1 gain is fitted on the other two S1 objects.
  - `oracle`: the gain is fitted on every reading of the pad (optimistic).
  - `none`: training-pad scale as labelled (reference only).

## Results (MAE TC, mean ± std over 5 seeds; reference 3 seeds)

| | glitch | all160 | reference: lab/capacity fpcnncoord (global scale, no blacklist) |
|---|---|---|---|
| **test, pad gain k=5** | **107.7 ± 1.3** | **107.8 ± 1.4** | 115.8 ± 1.1 |
| test, pad gain k=3 / k=10 | 109.5 / 106.4 | 109.8 / 106.7 | 118.9 / 114.5 |
| test, oracle gain | 104.7 ± 1.3 | 104.9 ± 1.3 | 112.3 ± 1.0 |
| 4-object mean, S1 cross-object gain | 118.9 ± 1.7 | 120.2 ± 2.9 | 127.9 ± 1.2 |
| test, uncalibrated (training scale) | 148.2 ± 1.1 | 142.6 ± 0.2 | 123.2 ± 2.7 |
| val (training pads) | 241.6 ± 5.6 | 233.6 ± 3.1 | 234.9 ± 1.9 |
| val S1 / S2 | 227 / 256 | 219 / 248 | 212 / 258 |

Per object at pad gain k=5 (MAE / SSIM):

| object | glitch | all160 | reference |
|---|---|---|---|
| Cuboid_TestXS (S1) | 177.1 / 0.66 | 176.2 / 0.66 | 193.6 / 0.60 |
| Sphere_TestG (S1) | 65.1 / 0.83 | 65.2 / 0.82 | 66.9 / 0.81 |
| Cylinder_TestM (S1) | 149.2 / 0.72 | 151.0 / 0.69 | 168.9 / 0.55 |
| Cylinder_TestButtons (S2) | 81.4 / 0.64 | 81.7 / 0.64 | 83.2 / 0.63 |

Fitted test-pad gains, relative to the training-pad scale: S1 ≈ 1.9 (glitch) / 1.75 (all160); S2 ≈ 0.22.

## Findings
1. **Calibrated, the two blacklists tie** (107.7 vs 107.8; the 4-object means and the cross-object numbers also overlap).
   - Dropping the 134 broad high rows neither helps nor hurts on the test pads.
   - The glitch blacklist (26 rows) is enough. It keeps the real high-load S2 responses.
2. **Uncalibrated test MAE is misleading here.** all160 looks 6 TC better (142.6 vs 148.2) only because removing the loud S2 rows lowers the S2 training scale, which happens to be closer to the quiet test S2 pad (Buttons 255 vs 273). After calibration the difference disappears.
3. **Validation favours all160 by 8 TC** (233.6 vs 241.6, about 1.3 combined std).
   - Val has no grip-235 rows and no flagged rows, so this is an indirect effect on grip-20 predictions.
   - It is a single object; weak evidence.
4. **The test pads need their own gain.**
   - Test S1 reads about 1.8–1.9× the training-S1 response; test S2 about 0.22× the training-S2 response. Same direction as the author's p99 estimates (1.39 and 0.27).
   - 3–5 readings per pad get within 3–5 TC of the oracle. A single reading is too noisy (k=1 is worse than k=3 by 11–12 TC).
   - With its own gain, Buttons finally beats the zero predictor (81 vs 112).
5. **Per-pad training scale + glitch removal + per-pad test gain beats the earlier best by about 8 TC** (107.7 vs 115.8 at k=5; oracle 104.7 vs 112.3). The gain comes from Cuboid (−17) and Cylinder M (−20).
   - This is not a controlled comparison: target scaling, the 26 dropped rows and the seed count all differ.
   - The uncalibrated ranking is reversed (148 vs 123). The global-scale model averages the two training pads, which is closer to test S1 by chance.

## Caveats
- One split seed. Val is one object at grip 20, so val says nothing about the grip-235 rows the blacklist targets.
- S2 test = one object (Buttons, 52 readings). Its k-shot calibration readings come from the same object it is scored on.
- The calibration draws pick random test readings, not controlled calibration contacts. The `s1_xobj` row is the stricter, new-object protocol for S1.
- SSIM uses a fixed data range (3,600 TC), so it also moves with amplitude: calibration changes it.

## Next
- To isolate per-pad target scaling from glitch removal, add a third arm: no blacklist + per-pad scale.
- Default from now on: glitch blacklist, per-pad target scale, and report test metrics with a per-test-pad gain (k=5) next to the uncalibrated number.
