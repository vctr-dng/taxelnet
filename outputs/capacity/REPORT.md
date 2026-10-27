# Capacity vs. underfitting on grasp-only data

Branch `lab/capacity` (from `lab/sensornorm`). Grasp-only training pairs (4,642 rows, `scripts/build_sensor_data.py`), global
target scaling, shape split seed 0 (train 4,334 rows / 13 shapes; **val = 308 rows of one object, Cuboid_G**), test = 358 rows,
300 epochs, best-val checkpoint, seeds 0-2. MLflow `taxelnet-capacity`. Files: `run_arms.sh`, `summarize.py`
(-> `results.md`, `results.json`), `scripts/capacity_eval.py` (train/val/test per pad/object, best and last epoch),
`scripts/memorize.py`, `scripts/noise_floor.py` (-> `noise_floor.json`), `memorize_c{16,64}/`, `calib/`.
All numbers: MAE in TC, mean ± std (population) over 3 seeds. Zero predictor: test 176, val ~261, train 279 (S1 135, S2 423).

## 1. Diagnosis

**Can cnn_v0 fit the training set? No (small widths), partly (wider).** All 4,642 grasp rows, weight decay 0, no val, no selection,
1,000 epochs cosine, lr 1e-3 (`memorize_c*/`):

| model | train MAE @100 / 300 / 1000 ep | S1 | S2 | clean rows only (no taxel > 3,000 TC) |
|---|---|---|---|---|
| cnn_v0 base 16 | 187 / 171 / **157.8** | 75 | 240 | 130.8 (zero predictor 249.9) |
| cnn_v0 base 64 | 166 / 142 / **117.2** | 52 | 182 | 93.5 |

The training loss is still falling slowly at 1,000 epochs, and the base-16 net removes only ~43% of the zero-predictor MAE
(158 vs 279). The standard runs (wd 1e-4, best-val checkpoint) have train MAE 192 (S1 89, S2 296), i.e. 69% of the zero predictor,
against val 232 (89%). Most of the training error is S2 (296 vs 89 for S1; the S2 target is also 3x larger) and the 160 glitch rows
(taxel > 3,000 TC, 152 of them S2) which cost about 900 TC each and are not predictable (see the audit).

**Capacity does fit the data, but fitting it does not generalise.** `cnn_feat` CoordConv big (base 32, 2 convs/block, GroupNorm)
reaches train MAE 68 at epoch 300 (see "Best vs last"), 130 at the best-val checkpoint; its val is 203 at best-val
and rises to 227 at the last epoch, test 124 -> 130. Wider plain nets: base 32/64 reduce last-epoch train MAE to 162/148 while val
at the last epoch gets worse (247/260 vs 236 for base 16). So more fitting past the best-val epoch buys no generalisation; the
model is memorising per-reading idiosyncrasies.

**Noise floor (`scripts/noise_floor.py`).** For each training row the nearest neighbour in PCA(30; 96% variance) space of the
scaled Z grid, same object and pad, other pose/force, has a real map that differs by a mean per-taxel L1 of **168 TC (S1 81,
S2 255; median 94)**, versus 279 for the zero predictor. The difference does not shrink for the closest neighbours (closest 10% by
relative input distance: 163; the rest 166-179), i.e. near-identical simulated deformations map to very different real maps.
If the neighbours were exact duplicates with iid noise, MAE floor ~ diff / sqrt(2) ~ **120 TC** (S1 ~57, S2 ~180). That is a
rough, input-based upper bound on the irreducible train error, not a clean replicate floor (the grasp set has no repeated
groups). Read against it: train 190 (standard run) is above it, train 68 (featbig, last) is far below it (memorisation),
val 203-232 is well above it, test 124-132 is below it only because test objects are easier (zero predictor 176).

**Verdict.** Mixed, and the lever is not capacity.
1. The default cnn_v0 (base 16) does underfit the training set (train MAE 192 vs a ~120 rough floor, and it cannot memorise).
   This confirms the leak-experiment observation (trained-on rows poorly fit).
2. Removing that underfit (base 64 memorisation 117, featbig train 68) does not carry over: val/test change little
   (base 64: val 231.5, test 130.2; featbig: val 203, test 124). The remaining val-train gap is a generalisation / input-information
   gap on top of a high noise level (nearest-neighbour inputs give targets 168 TC apart).
3. Small, real improvements come from architecture rather than width alone (below).

## 2. Arms (best-val checkpoint)

| arm | n | best ep | train | val | test | test (group macro) | calib k=5 | 4-grp mean | cos | aligned | peak hit | SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_v0 (control) | 3 | 77.7 ± 41.1 | 192.5 ± 8.6 | 231.7 ± 6.1 | 131.7 ± 0.8 | 131.7 ± 0.8 | 134.5 ± 1.0 | 150.5 ± 1.2 | 0.856 ± 0.009 | 109.7 ± 2.6 | 0.417 ± 0.035 | 0.631 ± 0.010 |
| cnn_v0 base 32 | 3 | 67.7 ± 17.0 | 182.9 ± 6.3 | 227.5 ± 2.6 | 126.4 ± 2.8 | 126.4 ± 2.8 | 129.7 ± 1.3 | 145.8 ± 3.6 | 0.863 ± 0.003 | 106.3 ± 0.6 | 0.444 ± 0.012 | 0.642 ± 0.006 |
| cnn_v0 base 64 | 3 | 52.0 ± 34.0 | 183.4 ± 10.7 | 231.5 ± 4.4 | 130.2 ± 3.4 | 130.2 ± 3.4 | 131.4 ± 0.9 | 149.8 ± 4.9 | 0.856 ± 0.008 | 108.1 ± 1.9 | 0.419 ± 0.036 | 0.636 ± 0.005 |
| cnn_feat CoordConv | 3 | 51.0 ± 36.0 | 193.0 ± 9.6 | 229.7 ± 6.2 | 127.8 ± 4.4 | 127.8 ± 4.4 | 127.0 ± 1.8 | 150.4 ± 6.3 | 0.872 ± 0.002 | 102.6 ± 0.9 | 0.431 ± 0.026 | 0.657 ± 0.003 |
| cnn_feat CoordConv big | 3 | 75.0 ± 31.0 | 130.1 ± 16.8 | 203.0 ± 5.0 | 124.1 ± 4.5 | 124.1 ± 4.5 | 121.6 ± 2.1 | 151.6 ± 4.6 | 0.915 ± 0.004 | 96.8 ± 2.6 | 0.541 ± 0.021 | 0.687 ± 0.016 |
| footprint (lr 1e-2) | 3 | 1.000 ± 0.000 | 333.1 ± 0.7 | 264.1 ± 0.8 | 179.0 ± 0.1 | 179.0 ± 0.1 | 144.7 ± 0.2 | 212.0 ± 0.2 | 0.790 ± 0.001 | 133.6 ± 0.6 | 0.295 ± 0.001 | 0.522 ± 0.001 |
| footprint_cnn | 3 | 176.0 ± 59.8 | 192.6 ± 7.4 | 231.5 ± 5.9 | 126.1 ± 1.8 | 126.1 ± 1.8 | 126.1 ± 1.5 | 146.1 ± 2.1 | 0.878 ± 0.006 | 104.2 ± 1.7 | 0.440 ± 0.016 | 0.670 ± 0.010 |
| footprint_cnn + CoordConv residual | 3 | 172.7 ± 59.6 | 188.6 ± 1.4 | 234.9 ± 1.9 | 123.2 ± 2.7 | 123.2 ± 2.7 | 125.3 ± 0.8 | 143.6 ± 3.4 | 0.890 ± 0.001 | 100.9 ± 0.5 | 0.432 ± 0.016 | 0.668 ± 0.011 |


## 3. Findings
- Control rerun: test 131.7 ± 0.8 (reference 130.9 ± 2.5), calib k=5 134.5, val 231.7; consistent.
- **Best arm on test: `footprint_cnn` + CoordConv residual, 123.2 ± 2.7** (per seed 123.4 / 119.8 / 126.4 vs control 132.1 / 130.7 / 132.5):
  -8.5 TC (6%), every seed below every control seed; Welch t ~ 4. Also best cos (0.890) among non-big arms. `footprint_cnn`
  alone: 126.1 ± 1.8 (-5.6). Gains are on S1 (118 vs 129) and Sphere (64 vs 74), Cyl M (168 vs 179); Cuboid XS stays ~190-200.
- `cnn_feat` CoordConv big: test 124.1 ± 4.5 (-7.6; one seed at 130.1, so the margin over seed noise is about 2-3 sigma), **best val
  by far (203.0 ± 5.0 vs 231.7)**, best calibrated k=5 (121.6), best cos/aligned/peak hit/SSIM (0.915 / 96.8 / 0.54 / 0.69), best Sphere
  (47). But it is the worst on Buttons (S2, 179 vs 146) and Cuboid XS is not better, and its last-epoch val worsens (227) as train goes to 68.
- Width alone (base 32: 126.4 ± 2.8, base 64: 130.2 ± 3.4) and CoordConv alone (127.8 ± 4.4) are within ~1-2 sigma of the control.
- `footprint` (physics only) cannot fit grasps (train 333, S1 309 vs zero 135): the indenter footprint model is the wrong
  functional form here. Its best-val epoch is epoch 1 in all seeds (test 179); the last epoch gives test 134 (val worsens 264 -> 284),
  so best-val selection fails for it. Pad-wise it is much worse on S2.
- Val is a single object (Cuboid_G, 308 rows) and S1 and S2 val errors differ by arm, so val-based ranking (featbig) and
  test-based ranking (footprint_cnn variants) disagree; do not over-read either.
- Test S2 is one object (Cylinder_TestButtons, whose label looks wrong, see sensornorm REPORT): the S2 test column is noisy and moves opposite to S1
  for big/CoordConv arms.

## 4. If capacity does not fix it: what the Z grid may be missing (not implemented)
Capacity helped the fit but not (much) the held-out error, and nearest-neighbour inputs predict targets ~168 TC off, so the input
looks like the limit. Candidates, in rough order of plausibility:
1. **Two-pad geometry.** A grasp squeezes the object between two pads; the real taxel map depends on the force split and on
   the opposing pad, but the model sees only one pad's Z deformation (S1/S2 given separately; nearest-input rows of the other pad differ by
   ~394 TC). Inputs of the opposing pad, or the object-pose relative to both pads, are missing.
2. **Tilt / contact angle.** Deg15-75 and V/H offsets change the load path and local pressure, but a Z map of the pad surface
   is a height-field only: shear and tangential contact forces, and the in-plane displacement, are not in it.
3. **Side-face / edge contact.** Contact on the side faces of the pad (or object edges) bulges the surface laterally; a pure
   Z field cannot represent overhangs or contact normals, which matters most for cuboid edges (Cuboid is the worst object, 190-205).
4. **Absolute force / contact-patch pressure.** Force is a group key, not an input; a map of Z does not fix the load (soft-body
   deformation saturates), so near-identical deformations can come from different forces and give different capacitance.
5. **Real-side artefacts**: S2 pad glitches (160 rows > 3,000 TC), per-sensor gain (sensorid agent), and taxel-level
   hysteresis are not recoverable from the input at all.

## 5. Caveats
- 3 seeds, one split seed; val is a single held-out object; 4 test objects, 1 of them the only S2 object with a suspicious label.
- The noise-floor figure is an approximation from near-duplicate inputs (neighbours differ in pose), not a replicate floor.
- Memorisation runs: one seed each, cosine lr, no early stopping, so "can't fit" is for 1,000 epochs at lr 1e-3; base-16 may fit
  more with longer training. Big/GroupNorm net fits in 300 epochs, so the optimisation/width story is partly about that net, not width.
- Train MAE for the arm table is evaluated at the best-val checkpoint (early in training for many arms: best epochs 50-170 of 300);
  last-epoch train MAE is also tabulated.
- Footprint arms use the old lab recipe (lr 1e-2 / physics_lr 1e-2, residual L1 0.05), not retuned for grasps.
- Calibration uses `scripts/calibrate_eval.py` (global-gain, k=5, object-macro), output in `calib/`.

## Detailed tables (from `results.md`)

#### Per pad

| arm | n | train S1 | train S2 | val S1 | val S2 | test S1 | test S2 |
|---|---|---|---|---|---|---|---|
| cnn_v0 (control) | 3 | 88.6 ± 4.6 | 296.4 ± 13.6 | 218.3 ± 10.1 | 245.0 ± 15.8 | 129.4 ± 0.8 | 145.5 ± 6.0 |
| cnn_v0 base 32 | 3 | 81.6 ± 4.2 | 284.2 ± 9.5 | 212.5 ± 0.7 | 242.4 ± 5.3 | 122.4 ± 1.3 | 149.5 ± 11.7 |
| cnn_v0 base 64 | 3 | 81.7 ± 4.3 | 285.1 ± 17.2 | 216.7 ± 1.7 | 246.3 ± 9.1 | 126.1 ± 0.4 | 154.1 ± 21.6 |
| cnn_feat CoordConv | 3 | 91.7 ± 6.5 | 294.4 ± 12.8 | 202.3 ± 3.8 | 257.1 ± 10.2 | 120.7 ± 2.6 | 169.7 ± 16.6 |
| cnn_feat CoordConv big | 3 | 61.0 ± 9.6 | 199.2 ± 24.2 | 200.4 ± 4.1 | 205.7 ± 13.9 | 114.8 ± 4.8 | 179.3 ± 7.9 |
| footprint (lr 1e-2) | 3 | 308.6 ± 1.1 | 357.5 ± 0.4 | 226.6 ± 0.7 | 301.6 ± 0.9 | 163.3 ± 0.2 | 271.3 ± 0.8 |
| footprint_cnn | 3 | 99.0 ± 12.3 | 286.3 ± 3.6 | 215.7 ± 3.5 | 247.3 ± 8.4 | 120.8 ± 1.0 | 156.8 ± 7.6 |
| footprint_cnn + CoordConv residual | 3 | 95.3 ± 4.6 | 281.8 ± 4.1 | 211.5 ± 5.7 | 258.3 ± 4.8 | 118.2 ± 0.9 | 152.8 ± 14.3 |

#### Test per object group

| arm | n | Cuboid XS | Sphere G | Cylinder M | Cylinder Buttons |
|---|---|---|---|---|---|
| cnn_v0 (control) | 3 | 203.5 ± 9.5 | 74.2 ± 3.2 | 178.8 ± 6.2 | 145.5 ± 6.0 |
| cnn_v0 base 32 | 3 | 193.0 ± 2.8 | 69.3 ± 2.4 | 171.3 ± 1.9 | 149.5 ± 11.7 |
| cnn_v0 base 64 | 3 | 196.9 ± 3.1 | 72.6 ± 2.2 | 175.5 ± 4.7 | 154.1 ± 21.6 |
| cnn_feat CoordConv | 3 | 191.7 ± 9.2 | 65.6 ± 0.3 | 174.6 ± 1.1 | 169.7 ± 16.6 |
| cnn_feat CoordConv big | 3 | 206.6 ± 9.8 | 47.2 ± 3.6 | 173.2 ± 2.8 | 179.3 ± 7.9 |
| footprint (lr 1e-2) | 3 | 215.2 ± 0.7 | 105.6 ± 0.3 | 255.8 ± 1.2 | 271.3 ± 0.8 |
| footprint_cnn | 3 | 190.6 ± 5.0 | 68.4 ± 1.5 | 168.7 ± 3.6 | 156.8 ± 7.6 |
| footprint_cnn + CoordConv residual | 3 | 190.0 ± 3.5 | 64.2 ± 2.9 | 167.5 ± 1.2 | 152.8 ± 14.3 |

#### Best vs last epoch

| arm | n | train best | train last | val best | val last | test best | test last |
|---|---|---|---|---|---|---|---|
| cnn_v0 (control) | 3 | 192.5 ± 8.6 | 177.6 ± 1.7 | 231.7 ± 6.1 | 236.3 ± 4.9 | 131.7 ± 0.8 | 134.5 ± 3.5 |
| cnn_v0 base 32 | 3 | 182.9 ± 6.3 | 161.9 ± 2.9 | 227.5 ± 2.6 | 247.0 ± 4.3 | 126.4 ± 2.8 | 143.3 ± 2.8 |
| cnn_v0 base 64 | 3 | 183.4 ± 10.7 | 148.4 ± 3.6 | 231.5 ± 4.4 | 259.6 ± 3.9 | 130.2 ± 3.4 | 155.1 ± 3.0 |
| cnn_feat CoordConv | 3 | 193.0 ± 9.6 | 165.8 ± 0.9 | 229.7 ± 6.2 | 257.1 ± 1.9 | 127.8 ± 4.4 | 145.8 ± 4.3 |
| cnn_feat CoordConv big | 3 | 130.1 ± 16.8 | 68.3 ± 1.1 | 203.0 ± 5.0 | 227.2 ± 9.9 | 124.1 ± 4.5 | 129.5 ± 5.0 |
| footprint (lr 1e-2) | 3 | 333.1 ± 0.7 | 222.3 ± 0.3 | 264.1 ± 0.8 | 284.4 ± 0.8 | 179.0 ± 0.1 | 134.0 ± 0.8 |
| footprint_cnn | 3 | 192.6 ± 7.4 | 185.8 ± 2.5 | 231.5 ± 5.9 | 233.1 ± 6.1 | 126.1 ± 1.8 | 125.5 ± 0.9 |
| footprint_cnn + CoordConv residual | 3 | 188.6 ± 1.4 | 182.4 ± 3.9 | 234.9 ± 1.9 | 237.3 ± 0.3 | 123.2 ± 2.7 | 125.2 ± 3.8 |

#### Train MAE per object

| arm | n | Cuboid_M | Cuboid_S | Cuboid_XS | Cylinder_G | Cylinder_M | Cylinder_S | Cylinder_XG | Cylinder_XS | Cylinder_XXG | Sphere_G | Sphere_M | Sphere_XG | Sphere_XXG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn_v0 (control) | 3 | 316.2 ± 16.7 | 206.8 ± 9.4 | 168.4 ± 9.5 | 194.1 ± 13.5 | 202.8 ± 9.3 | 153.3 ± 7.3 | 185.4 ± 10.8 | 183.9 ± 6.7 | 150.6 ± 8.5 | 152.4 ± 1.9 | 147.1 ± 2.8 | 178.0 ± 2.4 | 83.1 ± 6.5 |
| cnn_v0 base 32 | 3 | 309.5 ± 14.4 | 191.0 ± 4.0 | 154.4 ± 3.4 | 179.8 ± 8.0 | 191.0 ± 6.3 | 144.6 ± 5.0 | 174.4 ± 4.8 | 176.5 ± 4.5 | 141.8 ± 7.8 | 149.6 ± 5.4 | 145.5 ± 1.5 | 171.2 ± 7.4 | 73.8 ± 4.1 |
| cnn_v0 base 64 | 3 | 305.7 ± 22.2 | 195.4 ± 8.7 | 157.8 ± 9.6 | 181.0 ± 13.9 | 191.0 ± 11.0 | 145.3 ± 7.0 | 174.8 ± 11.7 | 176.5 ± 7.6 | 142.0 ± 15.2 | 147.9 ± 4.2 | 146.6 ± 3.0 | 169.6 ± 7.6 | 76.5 ± 6.8 |
| cnn_feat CoordConv | 3 | 317.9 ± 13.8 | 211.9 ± 10.9 | 173.2 ± 10.0 | 201.3 ± 14.2 | 204.6 ± 10.9 | 144.5 ± 11.3 | 192.1 ± 11.9 | 175.8 ± 9.6 | 148.1 ± 10.2 | 153.7 ± 3.1 | 145.4 ± 2.3 | 178.0 ± 2.2 | 82.9 ± 7.6 |
| cnn_feat CoordConv big | 3 | 203.2 ± 20.9 | 149.6 ± 28.3 | 102.7 ± 13.4 | 139.6 ± 13.0 | 150.8 ± 17.6 | 108.7 ± 14.2 | 135.7 ± 15.9 | 139.3 ± 15.3 | 73.2 ± 16.9 | 96.1 ± 9.9 | 108.6 ± 10.4 | 110.4 ± 14.2 | 42.5 ± 7.5 |
| footprint (lr 1e-2) | 3 | 415.5 ± 0.8 | 392.8 ± 1.4 | 374.8 ± 1.3 | 356.6 ± 0.8 | 352.6 ± 0.9 | 302.4 ± 1.0 | 386.9 ± 1.1 | 314.4 ± 0.9 | 207.2 ± 0.6 | 318.0 ± 0.7 | 292.4 ± 0.7 | 335.4 ± 0.7 | 135.8 ± 0.3 |
| footprint_cnn | 3 | 311.1 ± 6.6 | 216.2 ± 14.4 | 175.0 ± 12.0 | 193.2 ± 7.8 | 198.0 ± 6.5 | 148.4 ± 7.6 | 192.6 ± 12.0 | 179.6 ± 6.2 | 155.5 ± 6.6 | 153.2 ± 4.8 | 143.9 ± 2.1 | 174.1 ± 3.3 | 83.4 ± 4.2 |
| footprint_cnn + CoordConv residual | 3 | 300.4 ± 3.0 | 210.0 ± 5.5 | 168.6 ± 4.5 | 192.0 ± 2.3 | 197.0 ± 1.1 | 137.7 ± 2.3 | 194.0 ± 2.3 | 171.9 ± 2.3 | 161.5 ± 7.9 | 151.9 ± 3.5 | 142.6 ± 1.6 | 172.8 ± 3.3 | 85.9 ± 3.4 |

#### Val MAE per held-out object

| arm | n | Cuboid_G |
|---|---|---|
| cnn_v0 (control) | 3 | 231.7 ± 6.1 |
| cnn_v0 base 32 | 3 | 227.5 ± 2.6 |
| cnn_v0 base 64 | 3 | 231.5 ± 4.4 |
| cnn_feat CoordConv | 3 | 229.7 ± 6.2 |
| cnn_feat CoordConv big | 3 | 203.0 ± 5.0 |
| footprint (lr 1e-2) | 3 | 264.1 ± 0.8 |
| footprint_cnn | 3 | 231.5 ± 5.9 |
| footprint_cnn + CoordConv residual | 3 | 234.9 ± 1.9 |
