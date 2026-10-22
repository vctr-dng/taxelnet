# Leaner taxel head (2 taxel 3x3 convs with fewer channels)

Branch `lab/leaner` (from `lab/blacklist` plus its uncommitted code, copied), 2026-10-22. MLflow `taxelnet-taxelhead`.

**Setup:** cnn_v0 width 16 + 2 taxel 3×3 convs, `model.taxel_channels` 32 or 16. Same protocol as `lab/blacklist` outputs/taxelhead:
- grasp-only data, glitch blacklist, per-pad target scale;
- 5-fold object CV, one CNN per fold;
- MAE in TC, mean ± std over folds.

| taxel convs × channels | params | train MAE | val MAE (CV) | gap | uncal MAE | test-pad y_scale k5 | calib gain k5 | calib k20 | calib SSIM k20 |
|---|---|---|---|---|---|---|---|---|---|
| 0 (1×1 head) | 23k | 176.6 | 189.6 | 13 | 141.8 | 114.1 | 114.7 ± 2.2 | 113.6 | 0.710 |
| 1 × 64 | 60k | 155.3 | 179.6 | 24 | 128.4 | 102.7 | 102.3 ± 5.8 | 100.6 | 0.767 |
| 2 × 64 | 97k | 127.7 | **170.7** | 43 | **122.6** | 101.1 | 97.8 ± 11.3 | 96.2 | 0.767 |
| **2 × 32** | **51k** | 141.8 | 173.7 | 32 | 125.7 | 102.1 | **97.8 ± 8.1** | 96.3 | **0.773** |
| 2 × 16 | 35k | 149.9 | 175.1 | 25 | 125.5 | 102.1 | 102.8 ± 5.6 | 101.3 | 0.760 |

Per fold, val / calib gain k5:

| fold | 2 × 64 | 2 × 32 | 2 × 16 |
|---|---|---|---|
| f0 | 193 / 91 | 193 / 94 | 191 / 96 |
| f1 | 252 / 89 | 259 / 89 | 265 / 107 |
| f2 | 120 / 88 | 127 / 93 | 129 / 96 |
| f3 | 184 / 103 | 184 / 100 | 183 / 105 |
| f4 | 105 / 118 | 106 / 112 | 107 / 110 |

## Findings
1. **2 × 32 keeps the calibrated test performance of 2 × 64 with about half the parameters** (97.8 vs 97.8, 51k vs 97k). It is more stable across folds (std 8 vs 11), with a smaller train/val gap (32 vs 43) and the best calibrated SSIM (0.773).
2. **Validation is slightly worse with fewer channels:** +3 TC on average for 32 channels (two folds +7, three within 1), +4 for 16 (f1 +13). The 64-channel head still extracts a little more on held-out objects, at the cost of more fitting to training objects.
3. **16 channels is too lean on test** (calib k5 102.8, about the same as 1 × 64). All variants with 2 taxel convs beat 0 convs by 12–17 TC.
4. **Test-pad y_scale (k5) is about 102 TC for every 2-conv variant;** the fitted gain separates them more.

## Recommendation
**cnn_v0 width 16 + 2 taxel 3×3 convs with 32 channels** (`--model.taxel_convs 2 --model.taxel_channels 32`), 51k parameters: same calibrated test MAE as the 97k model, more stable, smaller gap. If validation MAE is the deciding metric, 2 × 64 is about 3 TC better on validation.
