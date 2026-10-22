# 3x3 convolutions on the 7x4 taxel map, test-pad y_scale (single standard CNN)

Branch `lab/blacklist`, 2026-10-22. MLflow `taxelnet-taxelhead`. Launcher `run_arms.sh`, table `results.md`.

## Setup
- **Model:** cnn_v0 width 16 (3 conv+pool blocks → adaptive pooling to 7×4, 64 channels).
  - **Change tested:** `model.taxel_convs` = 0 (original 1×1 head), 1 or 2 layers of 3×3 conv (64 ch) + ReLU on the 7×4 map before the 1×1 head. Extra parameters: 37k per layer.
- **Data:** grasp-only, glitch blacklist (26 rows), per-pad target scale.
- **Evaluation:** 5-fold object CV, fold k with seed k, one model per fold. No ensembling: every number is a single CNN, mean ± std over the 5 folds.
- **Test metrics, computed during training** (`calibration.unseen_pad_metrics`, 50 draws, same draws for both rescalings):
  - **uncalibrated:** training-pad y_scale;
  - **test-pad y_scale:** prediction × s_test / s_train, where s_test is the std of the test pad's real readings (the same statistic as the training scaler). Computed from k = 5 or 20 random readings of the pad (scored on the others), or from all of them (optimistic). Needs real readings only.
  - **calibrated:** one L1 gain per test pad, fitted on predicted vs real maps of k readings. Needs the simulated deformation of those readings.

## Results (MAE TC, SSIM; mean ± std over folds)

| taxel 3x3 convs | val MAE (CV) | uncal MAE | uncal SSIM | test-pad y_scale k5 | k20 | all | calib gain k5 | k20 | all | SSIM (calib k20) | 4-obj calib k5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (1x1 head) | 189.6 ± 55.7 | 141.8 ± 5.8 | 0.528 | 114.1 ± 2.2 | 114.0 | 112.4 | 114.7 ± 2.2 | 113.6 | 111.6 | 0.710 | 127.2 |
| 1 | 179.6 ± 58.2 | 128.4 ± 7.4 | 0.625 | 102.7 ± 4.1 | 102.2 | 101.2 | 102.3 ± 5.8 | 100.6 | 99.2 | 0.767 | 115.9 |
| **2** | **170.7 ± 53.1** | **122.6 ± 7.8** | **0.672** | **101.1 ± 8.5** | 101.3 | 101.0 | **97.8 ± 11.3** | 96.2 | 95.2 | 0.767 | 114.2 |
| ref: hybrid footprint_cnn+CoordConv (old split) | – | 148.2 | 0.490 | – | – | – | 108.3 | 106.7 | 104.7 | – | 119.1 |

Per fold (val / calib gain k5):

| fold (val objects) | 0 convs | 1 conv | 2 convs |
|---|---|---|---|
| f0 Cuboid_S, Cylinder_XG | 202 / 113 | 199 / 96 | 193 / 91 |
| f1 Cuboid_M, Cylinder_XXG | 282 / 118 | 273 / 108 | 252 / 89 |
| f2 Cylinder_XS, Sphere_M, Sphere_XG | 145 / 112 | 127 / 94 | 120 / 88 |
| f3 Cuboid_G, Cuboid_XS, Cylinder_G, Cylinder_M | 199 / 114 | 190 / 107 | 184 / 103 |
| f4 Cylinder_S, Sphere_G, Sphere_XXG | 120 / 116 | 109 / 106 | 105 / 118 |

## Findings
1. **3x3 convs at taxel resolution are the largest model gain so far.**
   - **Validation improves on every fold:** 0 → 1 conv by 4–18 TC; 1 → 2 by 4–21 TC.
   - **Calibrated test:** 114.7 → 102.3 → 97.8.
   - **Uncalibrated test:** 141.8 → 122.6.
   - **SSIM:** 0.53 → 0.67 (uncalibrated), 0.71 → 0.77 (calibrated).
   - The standard CNN with 2 taxel convs now beats the hybrid (97.8 vs 108.3, calibrated k=5).
   - Neighbouring taxels exchanging information is what the 1×1 head was missing.
2. **2 convs vs 1: better on average and on validation, but less stable on test.** Test std across folds is 11 vs 6. Fold 4 is worse with 2 convs (118 vs 106), the other four better (−4 to −19). Validation, which is what model selection uses, favours 2 on every fold.
3. **The test pad's own y_scale works as well as a fitted gain with 5 readings** (101.1 vs 97.8 for 2 convs, 102.7 vs 102.3 for 1, 114.1 vs 114.7 for 0).
   - It needs only real readings, no simulation of the calibration contacts.
   - With 20 readings the fitted gain pulls ahead a little (96.2 vs 101.3). It also improves more with k, while the std estimate barely changes.
   - **Fitted factors:** test S1 std / training S1 std = 1.35–1.61 (it varies by fold because the training scale does); test S2 = 0.22–0.24. The fitted gains are close for S2 (0.16–0.28) but smaller for S1 with 2 convs (1.05–1.30): the model then predicts louder for the test S1 pad on its own.
4. **Uncalibrated vs calibrated:** the gap shrinks with a better model (141.8 → 114.7 with 0 convs, 122.6 → 97.8 with 2), but rescaling per test pad still removes about 20 TC.

## Caveats
- Fold-to-fold spread is large (val 105–282 TC): compare models fold by fold, not by the mean ± std alone.
- k-reading versions take their readings from the test set at random, not from a controlled calibration session. "all" uses every test label.
- S2 test = one object (Buttons). At grip 20 the test S1 pad reads like training S1; the S1 factor comes mostly from grip 235 (Cylinder M).
- 3 or more taxel convs not tried. The trend (0 → 1 → 2) has not flattened on validation.

## Recommendation
- **Model:** cnn_v0 width 16 + 2 taxel 3×3 convs, one CNN.
- **Test scaling:** report uncalibrated, test-pad y_scale (k=5) and calibrated gain (k=5).
- **Deployment:** if only real readings of a new pad are available, use its own y_scale. If the calibration contacts can also be simulated, use the fitted gain.
