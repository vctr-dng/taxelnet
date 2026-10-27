# Train/test leakage in the Pushing_task tactile model: quantification report

Source material for a resume, portfolio entry or write-up. All figures come from the runs listed in the "Reproducibility" section. Wording that is safe to reuse verbatim is collected under "Resume material" at the end; the rest is background and caveats.

## 1. One-paragraph summary

A CNN that predicts real tactile-sensor readings (28 taxels) from simulated deformation grids was found to be evaluated with a random row-level train/test split. The dataset has several real readings per (path, force) condition, and all readings of a condition share one simulation input, so a random split puts the same input in train and test. I built a leak-free evaluation protocol (group-disjoint train/val/test, plus a deliberately leaky test set drawn from the same split), retrained the Pushing_task architecture across 3 splits × 3 training seeds (9 models), and quantified the effect with group-level bootstrap confidence intervals and a noise-floor analysis. On repeated-press conditions the leaky protocol cut the reported error by **61% (95% CI 52–68%)**, an absolute difference of **108 TC MAE (95% CI 73–141)**, and a **2.6× ratio** of clean to leaky error. About **78% (95% CI 67–85%)** of the model's genuine generalization error on unseen presses is hidden when a condition leaks into training.

## 2. Background and problem

- **Task:** predict the 28 readings of a real tactile sensor (a 7×4 taxel map, in tactile counts, TC) from a simulated 18×12 deformation height-field grid.
- **Original code:** `Pushing_task/CNN_Training/Training_CNN_06_01_v2.py` (Keras/TensorFlow). It builds one sample per real row: X is the first 18 simulation rows for that row's (force, path) key, y is the 28 real values. It then calls `sklearn.train_test_split` on the rows.
- **Why this leaks:** the real CSV has 13,653 rows over 9,796 (path, force) keys, but 682 keys have more than one real reading (4,539 rows, up to 54 readings per key). Repeated readings are the same press measured again, so they differ only by sensor noise and hysteresis. Their X is identical. A random row split can therefore put a key's X in train with one reading and in test with a repeat reading, and the network can score well by memorizing the key.
- **Size of the exposure in the original protocol:** with a 10% random split, 451 of 1,366 test rows (33%) had a key that also appeared in train.

## 3. Data

| Item | Value |
|---|---|
| Real readings (`Dataset_training_real.csv`) | 13,653 rows, 28 taxel columns (`data_0..data_27`) |
| Simulation rows (`Dataset_training_simulation.csv`) | 245,754 rows, 12 pixel columns (18 rows form one 18×12 grid per key) |
| Distinct (path, force) keys | 9,796 (the same set in real and simulation) |
| Keys with more than 1 real reading | 682 keys, 4,539 rows (max 54 readings) |
| Alignment | key = (force rounded to 2 decimals, stripped path); X = first 18 simulation rows of the key |

Behaviour kept from Pushing_task: only the first 18 simulation rows of a key are used. For keys with several simulated presses this means different real readings share one input. Kept in all arms so the split is the only variable.

## 4. Method

### 4.1 Model and training (PyTorch port of the Keras network)

- Conv(64, 2×2, no activation) → MaxPool → Conv(320) → Conv(512) → MaxPool → Conv(512) → MaxPool → global average pool → Dense(480)+ReLU → BatchNorm → Dropout(0.1) → Dense(352)+ReLU → Dense(28, linear). About 2.2 M parameters. Keras "same" padding semantics were reproduced (asymmetric padding for 2×2 kernels, ceil-mode pooling).
- Adam, lr 1e-4, batch 120, 600 epochs, L1-sum loss over the 28 outputs.
- Inputs divided by one scalar (the maximum over the training rows). Targets left in raw tactile counts, as in Pushing_task.
- The best epoch is chosen on validation loss only. Test sets are monitored every 10 epochs but never used for selection.

### 4.2 Fixed split (`scripts/make_pushing_split.py`, JSON file)

- Whole (path, force) groups are assigned to train / val / test_clean at 80/10/10 of the rows using the seeded, deficit-based `taxelnet.splits.allocate_groups`. Train, val and test_clean share no key.
- **test_leaky:** 1,366 rows taken out of multi-reading *train* groups (round-robin, one row per pass, always leaving at least one row of the group in train). Their keys are therefore also in train; this reproduces the Pushing_task leak. It is the same size as test_clean.
- Resulting row counts (split 0): train 9,555 (70%), val 1,366, test_clean 1,366, test_leaky 1,366. Train is 70%, not 80%, because the leaky rows are removed from the 80% train pool. Train and val are identical in every arm.
- The JSON stores row indices (into the real CSV), group lists, counts, seed and the CSV's SHA-256. The training script checks the row count against the CSV.
- Three split seeds (0, 1, 2) were generated; each uses a different group allocation.

### 4.3 Confound found and controlled

The first comparison (test_clean over all keys against test_leaky) looked inconclusive: test_leaky was only about as good as val and test_clean, and in early epochs it was worse. The cause was a group-mix confound: test_leaky can only contain keys with several readings, but test_clean is mostly single-reading keys, which are easier. Fix: score the clean sets restricted to multi-reading keys (`val_multi`, `test_clean_multi`), so both sides cover the same kind of press. That comparison is the headline result.

### 4.4 Metrics

- Micro (per reading) and macro (per (path, force) group) L1-sum loss, MAE per output and SSIM, with SSIM computed on the 7×4 map using one whole-map window and a fixed data range of 3,600 TC (`taxelnet.evaluation.global_ssim`). The SSIM reference is the paper in `papers/frobt-12-1639524.pdf`, which uses SSIM to compare synthetic and real tactile maps; it does not state its window or data range, so absolute SSIM values are not comparable to its 0.72/0.63.
- Prediction figures (sim input, target taxels, predicted taxels, residual; 6 fixed test readings, tagged "key in train" / "key unseen") logged to MLflow every 10 epochs.
- **Bootstrap:** resample (path, force) groups with replacement (4,000 draws), independently for the clean and leaky sets (they have disjoint readings). Per-reading errors are first averaged over the 3 training seeds of a split. The pooled interval averages the three splits within each draw. Row 0 of each draw array is the unresampled point estimate.
- **Noise floor:** for each reading, MAE against the per-taxel median of the *other* readings of the same key (leave-one-out, over the whole dataset). It approximates the error nothing could beat without information about the specific press. It is an approximation, not a strict lower bound.
- **Gap share of excess error:** (clean MAE − leaky MAE) ÷ (clean MAE − clean noise floor): the part of the genuine generalization error that leakage hides.

## 5. Results

### 5.1 Headline (9 models: 3 splits × 3 training seeds; 95% bootstrap CI over keys)

| Measure | Split 0 | Split 1 | Split 2 | Pooled |
|---|---|---|---|---|
| MAE, clean multi-reading keys (TC) | 199.7 | 166.0 | 162.8 | 176.2 [141, 208] |
| MAE, leaky (TC) | 67.2 | 66.6 | 69.9 | 67.9 [64, 72] |
| Gap, clean − leaky (TC) | 132.5 | 99.5 | 92.9 | 108.3 [73, 141] |
| Error reduction from leakage | 66% | 60% | 57% | 61% [52, 68] |
| Ratio clean ÷ leaky | 2.97 | 2.50 | 2.33 | 2.60 [2.06, 3.11] |
| SSIM gap (leaky − clean) | 0.34 | 0.19 | 0.20 | 0.24 [0.17, 0.31] |
| Noise floor, clean keys (TC) | 35.4 | 40.6 | 33.5 | 36.5 [30, 43] |
| Noise floor, leaky keys (TC) | 34.4 | 32.4 | 34.9 | 33.9 [31, 37] |
| Leaky MAE ÷ floor | 1.96 | 2.05 | 2.00 | 2.00 [1.86, 2.16] |
| Clean MAE ÷ floor | 5.65 | 4.09 | 4.86 | 4.83 [3.94, 5.81] |
| Share of excess error hidden by leak | 0.81 | 0.79 | 0.72 | 0.78 [0.67, 0.85] |

Per-run gap (MAE, TC; 9 runs): 129.2, 136.9, 131.5 | 107.4, 105.0, 86.0 | 87.9, 95.0, 95.8. Std over runs 19.5; seed-to-seed std within a split is only about 4 TC, so the split drives the variance, not the training seed.

### 5.2 Reference numbers from the first single-seed run (split 0, seed 0, best epoch 562)

| Set | Micro MAE | Micro SSIM |
|---|---|---|
| val (clean, all keys) | 68.7 | 0.875 |
| test_clean (all keys) | 85.8 | 0.815 |
| test_leaky | 68.2 | 0.853 |

Over all keys the leak is hidden by the single-reading keys (see 4.3). This is why the matched multi-reading comparison is needed.

### 5.3 Interpretation

- A model that has seen the same simulation grid in training, paired with a repeat reading, is about 2.0× the noise floor on that key. On a comparable press it has never seen, it is about 4.8× the floor. Most of the clean error is real generalization error.
- Using the leaky protocol would therefore make the model look about 2.6× better than it is on repeated-press conditions, and would make any comparison between models or fixes unreliable when the improvement is of memorization rather than generalization.
- Group-disjoint splits are required for a trustworthy comparison going forward.

## 6. Limitations (state these if the result is quoted)

1. **Scope of the effect:** the leaky set only contains multi-reading keys, so the figures describe the affected rows. The overall inflation of the original random 90/10 protocol is smaller, because many of its test rows are single-reading keys. It was not measured; an earlier attempt (random vs group arms) was stopped when the protocol was redesigned and its partial numbers are not reported.
2. **Sample size:** the clean multi-reading test set has 455 rows, hence wide intervals (the CI of the pooled gap spans roughly 73 to 141 TC). All nine runs agree on direction, and the lower end of the interval is still strongly positive.
3. **Splits and seeds:** 3 splits × 3 training seeds. The training-seed variance is small, the split-to-split variance is not.
4. **Not trained to convergence:** several runs were still improving at epoch 600 (best epochs 561–594). Absolute errors might fall with longer training; no reason to expect the sign of the gap to change.
5. **Train share:** train is 70% of rows, not 80% (see 4.2).
6. **Taxel order assumption:** the 28 outputs were reshaped to 7×4 row-major (as `tax1..tax28` in taxelnet); not verified against the original Pushing_task data.
7. **Port, not original code:** the PyTorch port reproduces the Keras architecture and hyperparameters but not bit-for-bit numerics. BatchNorm momentum/epsilon and initialisers were chosen to match Keras defaults approximately.

## 7. What was built

| File | Purpose |
|---|---|
| `scripts/make_pushing_split.py` | Seeded group-disjoint train/val/test_clean split plus leaky test; writes the JSON |
| `scripts/train_pushing_baseline.py` | PyTorch port of the Pushing_task CNN; trains from the JSON, evaluates val/test_clean/test_leaky (+ multi-reading subsets), picks the best epoch on val, logs metrics and figures to MLflow, saves checkpoint |
| `scripts/analyze_pushing_leakage.py` | Group bootstrap, noise floor, gap share; logs a run and writes `outputs/leakage_analysis.json` |
| `data/splits/pushing_split_s{0,1,2}.json` | The three fixed splits (git-ignored with `data/`) |

Reused from the repo: `taxelnet.splits.allocate_groups`, `taxelnet.evaluation.global_ssim` / `_micro_macro`, `taxelnet.plots.select_plot_samples`.

## 8. Reproducibility

```bash
uv run python scripts/make_pushing_split.py --seed {0,1,2} --out data/splits/pushing_split_s{0,1,2}.json
uv run python scripts/train_pushing_baseline.py --split_json data/splits/pushing_split_s0.json --seeds 0 1 2 \
    --experiment pushing-leakage-v3          # repeat for s1, s2
uv run python scripts/analyze_pushing_leakage.py --boot 4000
```

MLflow (experiment `pushing-leakage-v3`, tracking URI from `MLFLOW_TRACKING_URI`):

- Training runs, split 0: seed 0 `23dc88199e3f4dd6a8526546676c786e`, seed 1 `c4c8ae288d2f4a7b9a7a58eabf603e47`, seed 2 `ec952279883c4c779a69f4c84c043967`.
- Split 1: seed 0 `fd6a4a922b49488eb286815781f249b1`, seed 1 `9a5c1daeabb140d58538a1447c6c7311`, seed 2 `3a87924889dc4554918cb5a9ba3ddac3`.
- Split 2: seed 0 `759cb7aae2754a45aa927cc32fba9c0c`, seed 1 `8624ed1f2dce417395f459d4c80e8bed`, seed 2 `e67328670d5c4656a1e8e8433102db6e`.
- Final analysis run: `daf7437db58142e8a4376c821ca27b04` (run name `leakage-analysis`; earlier runs of the same name used one seed and should be ignored).
- Earlier single-run experiment `pushing-leakage-v2`: run `e64a5a9b95aa4129942ccc0f691167d2` (section 5.2).

The work was done on branch `training-pipeline` of the `taxelnet` repository. The split JSON files and checkpoints (`outputs/`) are git-ignored, and nothing was committed as part of this analysis.

## 9. Process notes (how the work was run)

- Dispatched independent training jobs to parallel subagents (2 for the first comparison, then 3 for the split-seed runs, then 3 more for the extra training seeds), each launching a script and reporting metrics, while the analysis tooling was written in parallel.
- Iterated on the experimental design as the evidence came in: replaced a random-vs-group arm comparison with one trained model scored on clean and leaky test sets, then found and removed the group-mix confound before drawing conclusions.
- Reported intervals and limitations rather than single point estimates.

## 10. Resume material

**Project title options:** "Quantifying train/test leakage in a sim-to-real tactile-sensing CNN"; "Leak-free evaluation protocol for tactile sensor prediction".

**Bullets (pick 2–4):**

- Diagnosed train/test leakage in a sim-to-real tactile-sensor CNN (repeated real readings shared one simulation input, so a random row split put 33% of test rows' inputs in training), and designed a group-disjoint train/val/test protocol plus a controlled leaky test set to measure it.
- Quantified the effect across 3 data splits × 3 training seeds with group-level bootstrap intervals: the leaky protocol reduced reported error by 61% (95% CI 52–68%) on repeated-press conditions (MAE 68 vs 176 tactile counts, 2.6× ratio, SSIM +0.24), hiding about 78% of the true generalization error.
- Identified and controlled a confound (easier single-reading keys diluting the comparison) by matching the group mix of the clean and leaky test sets, and benchmarked both against a leave-one-out noise floor (leaky error ≈ 2× the floor, clean ≈ 4.8×).
- Ported a Keras CNN to PyTorch and built a reproducible experiment pipeline (seeded split manifest with CSV checksum, MLflow tracking with per-epoch prediction figures, best-epoch selection on validation only), orchestrating parallel training runs with subagents.

**One-line version:** Showed that a random train/test split inflated a tactile-sensing CNN's apparent accuracy by 2.6× on repeated presses (error −61%, 95% CI 52–68%) and delivered a leak-free evaluation protocol with bootstrap confidence intervals.

**Skills evidenced:** PyTorch, experiment design, data-leakage detection, statistical resampling (group bootstrap), MLflow, reproducible pipelines, evaluation-metric design (micro/macro, SSIM, noise floor), parallel job orchestration.
