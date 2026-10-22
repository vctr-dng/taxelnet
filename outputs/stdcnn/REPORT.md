# Standard CNN, test-time calibration analyses, object k-fold, indenter/grasp transfer

Branch `lab/blacklist`, 2026-10-22. Goal: a deep-learning-only model with a standard CNN (no CoordConv, no
physics hybrid). All grasp runs use grasp-only data with the glitch blacklist (26 rows,
`outputs/blacklist/REPORT.md`) and a per-pad target scale. Test metrics use one gain per *test* pad fitted on k of
its readings (the test pads are not the training pads).

Files:
- `outputs/stdcnn/run_arms.sh`, `results.md`: standard CNN arms.
- `outputs/tier1/results.md|json` (`scripts/tier1_posthoc.py`): post-hoc analyses.
- `outputs/transfer/run_arms.sh`, `results.md`: transfer.
- MLflow experiments `taxelnet-stdcnn`, `taxelnet-transfer`.

## New infrastructure
- `data.split_strategy shape_fold` (`n_folds`, `fold`): k-fold over whole objects, balanced by reading count. Folds of the 14 grasp objects:
  - f0: Cuboid_S, Cylinder_XG;
  - f1: Cuboid_M, Cylinder_XXG;
  - f2: Cylinder_XS, Sphere_M, Sphere_XG;
  - f3: Cuboid_G, Cuboid_XS, Cylinder_G, Cylinder_M;
  - f4: Cylinder_S, Sphere_G, Sphere_XXG.
- `train.init_from BUNDLE`: warm start from a bundle's weights and input scaler; the target scaler is refitted.
- Training logs `test_padgain_k5_mae` (50 draws) and `test_padgain_oracle_mae` at every checkpoint and for the best model.
- Training artifacts: `plots/*/samples/<group>.png`, `ssim_by_group.png`, `plots/best_val/`.

## 1. Standard CNN (cnn_v0: 3 conv+pool blocks, adaptive pooling to 7x4, 1x1 head)
MAE TC, mean ± std. "Old split" = shape split seed 0 (val Cuboid_G only), seeds 0–4. "5-fold" = fold k with seed k.

| arm | val MAE | test, uncalibrated | **test, pad gain k=5** | 4-object mean k=5 | test, oracle gain |
|---|---|---|---|---|---|
| cnn_v0 width 16, old split | 245.6 ± 2.6 | 146.8 | **116.3 ± 1.6** | 128.3 | 113.4 |
| cnn_v0 width 32, old split | 247.8 ± 2.0 | 146.0 | 117.8 ± 0.5 | 129.8 | 114.8 |
| cnn_v0 width 16, 5-fold | 188.3 (CV) | 140.1 | **114.2 ± 1.8** | 126.6 | 111.0 |
| cnn_v0 width 32, 5-fold | 188.3 (CV) | 139.5 | 113.8 ± 2.6 | 126.4 | 110.6 |
| reference: footprint_cnn + CoordConv (hybrid), old split | 241.6 | 148.2 | 108.3 ± 1.6 | 119.1 | 104.7 |

- **Width 16 and 32 tie** (paired per fold too). Width 32 peaks very early (best epoch ≈ 38) and overfits.
- **The standard CNN is about 8 TC behind the hybrid** (116 vs 108). The gap is mostly Cuboid and Cylinder M.
- **5-fold models score 2 TC better on test** (114.2 vs 116.3) although each trains on fewer objects. Choosing the checkpoint on 3–4 held-out objects is more reliable than on Cuboid_G alone.
- **Validation per fold is very uneven:** 120–278 TC depending on the held-out objects (f1 with Cuboid_M is hardest, f4 easiest). A single held-out object can't rank models; the 5-fold mean, compared fold by fold, can.

## 2. Tier 1: post-hoc, no retraining (`outputs/tier1/`)
Test MAE TC, per-test-pad fits on k readings (50 draws). The two right columns are 5-fold averages.

| method | cnn16 old split k5 / k20 | cnn16 5-fold k5 / k20 | hybrid ref k5 / k20 |
|---|---|---|---|
| one gain per pad (reference) | 116.3 / 115.4 | 114.2 / 113.0 | 108.3 / 106.7 |
| power curve `g·p^γ` | 116.4 / 114.1 | 113.8 / 111.1 | 108.2 / 105.7 |
| per-taxel gain + offset (shrunk, τ=3) | 114.8 / 110.9 | 112.5 / 108.1 | 106.4 / 101.0 |
| per-taxel gain only (no offset, any τ) | ≥ 116 | ≥ 114 | ≥ 108 |
| per-row gain (7 per pad) | no gain at k ≤ 20 | – | – |
| ensemble of 5 models + one gain | 114.4 / 113.5 | 111.7 / 110.8 | 107.0 / 105.1 |
| **ensemble + per-taxel gain + offset** | **112.8 / 108.8** | **110.1 / 105.4** | 105.0 / 99.1 |

- **Ensembling the 5 models: −2 to −2.5 TC**, consistently.
- **Power curve:** nothing at k=5; −1 to −2 TC with 10–20 readings. Amplitude compression is real but small once the gain is fitted.
- **Per-taxel gain + offset:** −1.5 TC at k=5, −4.5 at k=20. The *offsets* carry the benefit: gain-only per-taxel or per-row fits never beat one gain. The test pads seem to have taxel baseline offsets that a gain can't remove. Shrinkage toward the pad fit is required (unshrunk at k=5: 170+).
- **Grid registration is fine.** Shifting the predicted map by ±1 row or column worsens every object (113 → 151–175 TC), so there is no sim/real offset to correct. Sphere's "peak one row lower" cases are per-reading, not systematic.
- **Best standard-CNN result:** 5-fold ensemble + per-taxel gain/offset on 20 readings per pad = **105.4 TC**; 110.1 with 5 readings.

## 3. Transfer between indenter and grasp data (cnn_v0 width 16)
Pre-training: 300 epochs, lr 1e-3. Fine-tuning: 150 epochs, lr 3e-4. The input scaler is reused, the target scaler refitted.

| arm | indenter val (held-out shapes) | grasp val | grasp test, pad gain k=5 | 4-object mean k=5 |
|---|---|---|---|---|
| indenter only | **108.2 ± 0.2** | – | 144.7 ± 1.1 | 158.9 |
| grasp → indenter (fine-tuned on indenter) | 109.5 ± 1.2 | – | 140.5 ± 1.6 | 153.7 |
| grasp only (old split) | – | 245.6 | **116.3 ± 1.6** | 128.3 |
| indenter → grasp (old split) | – | 247.5 | 119.5 ± 0.4 | 131.4 |
| grasp only (5-fold) | – | 188.3 | **114.2 ± 1.8** | 126.6 |
| indenter → grasp (5-fold) | – | 199.6 | 118.3 ± 1.4 | 130.7 |

- **Indenter pre-training hurts grasp: +3 to +4 TC on test, worse validation on every fold** (paired: +4 to +17 TC). After fine-tuning, the early indenter features (best indenter epoch 15 on average) give no head start for grasps.
- **Grasp pre-training doesn't help the indenter task:** 109.5 vs 108.2. It does keep a little grasp skill after indenter fine-tuning (140.5 vs 144.7), but far below grasp-only training.
- **Conclusion:** the two domains don't share enough to help each other with this CNN. With merged training (earlier: 151 vs 131) and both transfer directions now all negative or neutral, the grasp model should be trained on grasp data alone.

## Caveat on the test-pad gain (from the side analysis)
At grip 20, test S1 reads like training S1: per-reading standard deviation 370 vs 376 TC. The "test S1 ≈ 1.9× louder" gain comes from grip 235, which on the test side is only Cylinder M (54 readings). So the pad gain partly corrects a force-dependent response, not only a sensor difference. A calibration protocol for new pads should cover the same force levels as deployment.

## Recommended standard-CNN setup
- cnn_v0 width 16; grasp-only data, glitch blacklist, per-pad target scale.
- Checkpoint selection on 5-fold object CV, then the 5 fold models used as an ensemble.
- At deployment, per new pad: one gain from about 5 readings, or per-taxel gain + offset (shrunk) with 20+ readings.
- Expected test MAE: 110 (5 readings) to 105 (20 readings), against 99–105 for the same pipeline with the hybrid model.

## Next steps, standard CNN only
1. **3×3 convolutions at taxel resolution after the pooling**, instead of the 1×1 head, so neighbouring taxels exchange information (the author's architecture C does this).
2. Train the 5-fold ensemble with more seeds per fold, if ensembling keeps paying.
3. A force-aware calibration protocol for new pads, since the pad gain depends on grip force (see caveat).
