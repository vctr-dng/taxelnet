# Audit: merged indenter + grasp training data (2026-10-15)

Worktree branch `worktree-grasp-merge` (based on `lab/p2b`). Merged data in
`data/newton/v1/merged/` (`scripts/merge_training_data.py`, see `merged_data_README.md`).
The audit script is `audit.py` and its figures and `summary.json` are in `out/`. The
baseline runs are summarised by `summarize_runs.py` in `out/baseline_runs.{md,json}`; bundles
are in `merged_baseline/` and in MLflow experiment `taxelnet-v0-merged`.

## 1. Merged data

| | Rows | Notes |
|---|---|---|
| Indenter (`Dataset_Indenter_*`) | 12,730 | unchanged, datasets 01–06 |
| Grasp (`Dataset_Grasp_*_qtac100`) | 4,642 | 2,321 grasps × pads S1/S2, 14 objects, `dataset = grasp` |
| **Merged training pair** | **17,372** | 13,879 `(force, Path)` groups |
| Test (`*_testing`, unchanged) | 358 | 4 objects `*_Test*` |

**Grasp training content:**
- **Objects:** Cuboid G/M/S/XS, Cylinder G/M/S/XG/XS/XXG, Sphere G/M/XG/XXG.
- **Force labels:** 20 on 886 rows and 235 on 3,756 rows, so 81% of the grasp rows are at max grip.
- **Tilts:** 0°–90° in 15° steps.
- **Tokens:** A/B (3,062 / 1,580).

**Integrity:**
- The strict loader accepts the merged pair.
- No simulation differs between rows of one key.
- No duplicate grasp simulations.
- No key, name or object shared with the test set.
- The test set's 20° tilt (35 rows) does not occur in training, which has 15° and 30°. This is minor.

**Split:**
- With `split_strategy=shape`, val holds out 7 shapes, 3 of them grasp objects (Cuboid_G, Sphere_G, Sphere_M), for 1,702 rows, 752 of them grasp.
- Val numbers are therefore not comparable to the indenter-only runs. Test numbers are.
- With `split_strategy=group`, a grasp's S1 and S2 pads may end up in different partitions.

## 2. Findings, most important first

1. **The grasp simulations are consistent with the measured grip forces** (high confidence).
   - With the indenter law (Σ deformation = 7.97·F − 4.15, r = 0.998), the training grasp simulations imply a median of **23 N** for label 20 (p10–p90: 13–36) and **98 N** for label 235 (88–108).
   - This matches the author's measurements of ≈20 N and ≈90 N. The test groups fit too: about 26 N for the "20" groups and 87–101 N for the "235" groups.

2. **Pads S1 and S2 are different sensors with very different responses** (high confidence).
   - Same grasps, similar simulated load, median Σ real signal:

     | | 20 N | ≈90 N |
     |---|---|---|
     | S1 | 4,613 TC | 2,289 TC |
     | S2 | 6,818 TC | 8,476 TC |

   - The median S2/S1 ratio is e^1.24 ≈ 3.4×. Relative to the indenter law, S1 reads 0.11× and S2 0.43×.
   - S2's 99th-percentile taxel reaches 7,735 TC, while S1 and the indenter sensor stay below about 2,700 TC.
   - The model input does not encode which pad is measured, so this gain difference cannot be learned from the input. It becomes irreducible error unless a sensor id is given to the model.
   - The test set is S1 except Cylinder_TestButtons, the only S2 group.

3. **At max grip, the real signal saturates or even drops.**
   - S1 gives a lower total signal at ≈90 N (2,289 TC) than at 20 N (4,613 TC).
   - In the training grasp data, the log-log relationship between Σ deformation and Σ signal is flat to negative: slope −0.19, against +0.82 for the indenter data.
   - The indenter data, at 51 N maximum, cannot teach this regime. The grasp data now covers it.

4. **Extreme taxel values in the grasp training data** (high confidence on the counts).
   - 306 grasp rows have a taxel above 3,000 TC: 71 above 5,000 and 16 above 10,000, up to **52,158 TC**.
   - All are max-grip rows, 298 of them S2, mostly cylinders and cuboids at 60–75° tilt.
   - The indenter maximum is 2,711 TC. The values near 50k look like overflow or glitches.
   - These rows enter training unchanged. They should be checked with the author, and possibly excluded or clipped.

5. **Strong negative values in the grasp data.**
   - 13.5% of off-contact taxels in grasp training are below −50 TC (minimum −4,921), against 4.7% in the indenter data.
   - The negative offsets seen on the test cylinders are therefore a property of the grasp recordings, and training now covers them.

6. **The test set is more consistent with grasp training than with the indenter data, but not fully.**
   - **Cuboid_TestXS:** nearest training neighbours are Cuboid_G; real signal ratio to them is **0.95**. Consistent.
   - **Sphere_TestG:** neighbours are Sphere_XXG; ratio **0.65** (it was 0.37 against indenter neighbours).
   - **Cylinder_TestM:** neighbours are training cylinders; ratio **1.88**, so the test is louder than training.
   - **Cylinder_TestButtons (S2):**
     - Its nearest-neighbour distance is 0.83, against 0.18 for a training grasp object relative to the other objects. It is out of distribution, probably because of the button relief.
     - Its real-signal ratio to its neighbours is **0.28**.
     - It also reads 0.27× the median of training S2 rows at max grip.

7. **Simulated patterns match real ones less well for grasps.** The median cosine between the pooled simulation and the real map is 0.92 for the indenter data, against 0.56–0.85 for training grasp objects (Cuboid_XS 0.56, Cuboid_S 0.64, Cylinder_XG 0.64). Tilted grasps and the two-pad geometry reproduce less faithfully.

8. **Small simulation/real mismatches.** 8 grasp Cuboid rows have almost no simulated contact (Σ < 5 mm) but more than 3,000 TC real signal, probably flat-face contacts the simulation missed. Under a looser cut (Σ < 50 mm) there are also 20 indenter rows and 5 test Cuboid rows.

9. **The `std` columns are temporal noise** (confirmed by the author): median 9.4 TC in training, 9.3 TC in test.

## 3. Baseline CNN (`cnn_v0`, default flags, 300 epochs) on the merged data

MAE in TC; test metrics on `best.pt`. Zero-predictor references: test 176, Cuboid 294, Sphere 106, Cylinder_TestM 262, Cylinder_TestButtons 112.

| run | train data | best ep | val MAE | val indenter / grasp | test MAE | 4-group mean | calib k=5 | Cuboid | Sphere | Cyl TestM | Cyl Buttons | cos | aligned | peak hit | amp log-ratio | SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0-shape-s0 | indenter+grasp | 62 | 144.8 | 112.8 / 186.2 | 153.8 | 173.9 | 138.7 | 190.4 | 104.9 | 180.0 | 220.4 | .803 | 119.8 | .352 | −.017 | .558 |
| m0-shape-s1 | indenter+grasp | 117 | 146.1 | 114.4 / 186.9 | 150.1 | 176.0 | 140.0 | 186.3 | 90.5 | 190.5 | 236.8 | .817 | 117.6 | .399 | −.171 | .543 |
| m0-shape-s2 | indenter+grasp | 102 | 145.7 | 112.6 / 188.6 | 148.8 | 170.0 | 133.8 | 183.3 | 98.9 | 169.1 | 228.6 | .824 | 114.8 | .358 | −.012 | .587 |
| m0-group-s0 | indenter+grasp | 289 | 127.1 | 102.5 / 210.6 | 147.1 | 170.5 | 133.4 | 188.8 | 90.8 | 170.4 | 231.9 | .851 | 110.1 | .436 | −.066 | .580 |
| p0-shape (3 seeds, mean) | indenter only | 17 | 108.5 | – | 227.3 | 276.4 | 148.3 | 203.4 | 141.5 | 400.9 | 359.8 | .715 | 132.4 | .219 | .407 | .449 |

**Shape split, merged vs indenter-only, mean of 3 seeds each:**
- **Test MAE** drops from 227 to **151**. This is the first V0 below the zero predictor (176).
- **4-group mean** drops from 276 to **173**.
- **Calibrated k=5 MAE** drops from 148 to **137**.
- **Cylinder_TestM** drops from 401 to **180**.
- **Cylinder_TestButtons** drops from 360 to **229**. It is still worse than predicting zero (112).
- **Sphere** drops from 141 to **98**, now at about the zero predictor (106).
- **Cuboid** drops from 203 to **187**.
- **Pattern metrics** improve: cosine 0.715 → **0.815**, aligned MAE 132 → **117**, peak hit 0.22 → **0.37**, SSIM 0.45 → **0.56**.
- **Amplitude bias is gone:** the amplitude log-ratio goes from +0.41 to **−0.07**.

**Val:**
- The held-out indenter shapes give 112–114, about the same as indenter-only training (108.5 on different shapes). Adding grasp data does not hurt the indenter domain.
- The held-out grasp objects give **187**, much higher.
- The best epoch moves from about 17 to 60–120, so the held-out-shape val no longer peaks very early.

## 4. Recommendations

1. **Give the model the sensor identity.** A pad or sensor id (indenter sensor, S1, S2) as input, or a per-sensor gain/offset head, would account for the 3.4× S2/S1 gain. Without it, that gain is unexplainable error. The few-shot calibration already does this at evaluation time (calib k=5 = 137).
2. **Resolve the extreme values with the author** (306 rows above 3,000 TC, up to 52k). Then exclude them, clip them, or add a robust-loss ablation, and retrain.
3. **Re-run the lab winners on the merged data:** the footprint/CNN hybrid, CoordConv, and the hybrid with CoordConv. The force-saturation regime needs a saturating response, which the footprint power law and the bounded head can represent.
4. **Report val per source** (indenter vs grasp) and keep held-out grasp objects in val. The pooled val MAE now mixes two very different difficulties (113 vs 187).
5. **Ask the author** what the Buttons object is, why the S2 test recordings are so much quieter than training S2, and whether S1 and S2 are fixed physical pads, the same across all sessions.
