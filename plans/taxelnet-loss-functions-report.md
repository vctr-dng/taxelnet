# Taxelnet Loss Functions — V0, V2, V3

## Goal

Choose and justify the training loss and evaluation metrics for three model
versions of the Taxelnet tactile surrogate (`taxelnet/plans/second.md`):

- **V0**: deterministic spatial CNN.
- **V2**: mean plus per-taxel aleatoric scale.
- **V3**: correlated conditional tactile samples.

Sources:

- De la Cruz Sánchez & Roberge, *A hybrid elastic-hyperelastic approach for
  simulating soft tactile sensors*, Front. Robot. AI 12:1639524 (2025).
- De la Cruz Sánchez, Kwiatkowski & Roberge, *Tactile Contact Patterns for
  Robotic Grasping: A Dataset of Real and Simulated Data*, ICCRT 2025.
- Read-only statistics computed on
  `taxelnet/data/raw/Dataset_training_real.csv` (2026-10-02).

The resulting specification has been folded into `taxelnet/plans/second.md`
(sections *Group Weighting*, *Training-Side Normalization*, *Evaluation*,
*Regression Gates*, *Real-Sensor Variation Evidence* and the V0/V2/V3 roadmap).

---

## What the Papers Establish

- **Original loss.** Frontiers Eq. 1 is summed L1 over the 28 taxels, with a
  dense 570 → 28 head, trained on about 13k indenter maps (80/20 split). The
  input there was a 33 × 57 deformation map; Taxelnet's input is 18 × 12.
- **Evaluation used SSIM.** 0.81 (Abaqus) and 0.82 (Isaac) on unseen indenters;
  0.72 and 0.63 on the 12 grasped objects. The authors state that accurate
  excitation patterns matter more than exact taxel values.
- **Trial-to-trial variation is known but ignored by the loss.** 50 trials per
  gripper–object position show varying taxel counts, attributed to sensor
  hysteresis (Kwiatkowski et al., 2022). Those readings are compared against a
  single deterministic synthetic map.
- **Baseline removal.** The mean of the first 50 rest samples is subtracted
  (ICCRT §III-B1). This explains negative values; drift after subtraction shows
  up as a whole-map offset.

---

## Data Facts That Decide the Loss

| Quantity | Value |
|---|---|
| Readings / raw `(force, path)` groups | 13,653 / 9,796 |
| Singleton groups | 9,114 (93%) |
| Repeated groups | 682 groups, 4,539 readings (33%); largest 54; CircleG = 47% of these readings |
| Target range | −888 to 2,711 TC; 23% negative; 39% with \|y\| < 30 |
| Residual scale vs signal | median \|residual\| 8 → 22 TC as \|group mean\| goes from < 9 to > 374 TC, then levels off |
| Noise floor (MAE vs replicate group median, contact) | micro 29 TC, macro 21 TC |
| Trivial predictors | all-zero MAE 286 TC; global per-taxel median 271 TC |

Replicate residuals (reading minus its group mean, groups with ≥ 2 readings):

| Quantity | No contact (force = 0) | Contact (force > 0) |
|---|---|---|
| Repeated groups / readings | 24 / 240 | 658 / 4,299 |
| Residual std | 11 TC | 67 TC |
| std / median \|residual\| (Gaussian 1.48, Laplace 2.04) | 2.0 | 4.9 |
| Mean cross-taxel correlation | 0.05 | 0.47 |
| Variance in top two eigenmodes | 29% | 84% |
| Variance in a whole-map common offset | 7% | 41% |

Consequences:

1. **Heavy tails** argue against L2 as a training loss.
2. **Noise depends on contact**: about 6× larger under contact.
3. **Contact noise is correlated** across taxels and close to low-rank, largely
   a whole-map offset.
4. **Most groups are singletons** because grouping uses the exact raw force.
   Within one path, singletons are effectively repeated trials at slightly
   different forces.
5. **Achievable V0 range**: MAE between ~21–29 TC (noise floor) and ~270 TC
   (trivial).

Caveat: pooling groups with different noise scales inflates the std/median
ratio. Recheck tails on residuals scaled by V2's predicted scale, and recompute
all statistics for every new dataset.

---

## V0 — Deterministic Spatial CNN

**Loss: group-balanced per-taxel L1.**

```text
L_V0 = mean over groups g [ mean over readings j in g [ mean over taxels t
         |mu(x_g)_t - y_{j,t}| ] ]
```

- **Estimand.** L1 predicts the conditional per-taxel *median*. Heavy tails
  justify L1 over L2; MSE/RMSE remain evaluation metrics. Contact-position
  jitter blurs peaks under any point loss — a limit of point prediction and part
  of V3's motivation.
- **Legacy loss.** Summed L1 = 28 × MAE. Adam is roughly invariant to that
  constant, so they train alike. Keep summed L1 only for the
  `LegacyDenseHeadCNN` reproduction.
- **Group weighting.** Loss per reading → mean within group → mean across
  groups. Never fit to a group-mean target (that changes the estimand). Balancing
  mostly down-weights round-force trials of a few indenters (up to 54×). Log the
  effective weight share per path; a tempered weight `n_g ** alpha` is an
  optional ablation.
- **Units.** L1 on `per_taxel_zscore` targets equals raw L1 weighted by
  `1 / sigma_t`. Compute every gate in raw tactile counts.
- **Optimization.** The L1 gradient does not shrink near the optimum — use a
  learning-rate schedule. Huber with δ ≈ 10–20 TC is an ablation only.
- **No SSIM in the loss.** Windowed SSIM is degenerate on 7 × 4 maps and
  unstable near zero. Report global SSIM with a fixed `data_range` as a metric.
- **Reference points in `metrics.json`**: replicate noise floor and all-zero
  predictor MAE.

---

## V2 — Mean Plus Aleatoric Scale

**Loss: Laplace NLL with β = 1** (Seitzer et al., ICLR 2022), group-balanced:

```text
l(y, mu, b) = sg(b) * ( |y - mu| / b + log(b) )     # sg = stop-gradient
b           = b_min + softplus(raw_scale(x))
```

- **Why β = 1.** The `mu` gradient is exactly `sign(mu - y)` — identical to
  V0's L1 — so V0's gates still apply to `mu`. The `b` gradient is
  `1 - |y - mu| / b`, so `b` converges to E|y − mu|, the Laplace MLE. The plain
  NLL divides the `mu` gradient by `b`, starving high-noise regions (here ~6×
  noisier). Warm-start `mu` from V0.
- **Wording.** Under Laplace, `mu` is the conditional median, not the expected
  map.
- **Scale parametrization.** A single global per-taxel scale times
  `exp(delta)` with strong shrinkage cannot cover the 6× regime gap
  (|δ| ≈ 1.8). Use a physical baseline:

  ```text
  b0(x, t) = b_floor(t) + kappa(t) * g(|mu(x, t)|)    # g saturating
  b(x, t)  = b0(x, t) * exp(delta(x, t))              # delta shrunk to 0
  ```

  Fit `b0` from repeated groups but treat it as a prior (47% CircleG). Set
  `b_min` to a few TC.
- **Calibration.** Singletons do train `b(x)`, but `b` can absorb model misfit
  and sim-to-real bias, and training residuals are optimistic. Fit a per-taxel
  multiplier on `b` on the validation split; report test coverage.
- **Tail check.** `|y - mu| / b` should follow Exp(1). If tails stay heavy,
  switch to a Student-t likelihood with a learned global ν, keeping β = 1.
- **Scaled targets.** Raw NLL = scaled NLL + Σ_t log σ_t.
- **Samples.** V2 samples are independent per taxel and look like
  salt-and-pepper noise given the 0.47 contact correlation. Export `mu` and
  intervals to IsaacLab, not V2 samples.

Metrics:

- Laplace CRPS: `|y - mu| + b * exp(-|y - mu| / b) - 3b/4`. A point forecast's
  CRPS equals its MAE, so V0 and V2 share a scale.
- Central interval at level p: `mu ± b * ln(1 / (1 - p))`, at 50/80/90/95%.
- PIT histogram; NLL and coverage on held-out repeated groups, split by
  contact.

---

## V3 — Correlated Conditional Samples

The trigger (spatially correlated residuals) is already met: correlation 0.47,
84% in two modes, 41% common offset. Multimodality is untested. In every V3
variant, keep `mu` trained by L1 (frozen or decoupled).

### V3a — Low-Rank Plus Diagonal Gaussian (first)

```text
y | x ~ Normal(mu(x), W(x) W(x)^T + diag(d(x))),   K = 2-4
W(x)  = B * diag(a(x))   # B: fixed 28 x K basis from PCA of replicate residuals
```

- Exact group-balanced NLL via `torch.distributions.LowRankMultivariateNormal`.
- A linear-decoder CVAE with exact likelihood: no ELBO gap, no posterior
  collapse.
- Sampling `mu + W z + sqrt(d) * eps` exports to ONNX with noise inputs.

### V3b — Nonlinear Generator (only if V3a loses or residuals are multimodal)

**Preferred: energy-score generator** (engression; Shen & Meinshausen):

```text
y_hat_j = g(x, eps_j),  j = 1..m,  m >= 2
ES      = mean_j ||y_hat_j - y|| - 1 / (2 m (m - 1)) * sum_{j != k} ||y_hat_j - y_hat_k||
```

- Proper scoring rule; training loss = comparison metric.
- No encoder → no leakage of misfit into a latent, no posterior collapse.
- No likelihood family to choose; ONNX export with a noise input.
- For repeated groups, average over all readings in the group, then balance.
- ES is weakly sensitive to correlation → add a variogram-score term.

**Alternative: CVAE** (Sohn et al., NeurIPS 2015):

```text
-E_q[log p(y | x, z)] + beta * KL(q(z | x, y) || p(z | x))
```

Safeguards: Laplace decoder with V2's learned scale (fixed-scale L1 is an
implicit β); free bits or KL annealing with per-dimension KL monitoring;
check that the encoder is not storing sim-to-real misfit in `z`; Sohn's hybrid
objective for posterior-vs-prior `z` mismatch; NLL only as an importance-weighted
bound.

---

## Unified Metric Ladder

| Metric | V0 | V2 | V3a | V3b |
|---|---|---|---|---|
| MAE / RMSE of point (`mu` or sample median) | ✓ | ✓ | ✓ | ✓ |
| Per-taxel CRPS | = MAE | closed form | samples | samples |
| NLL (raw units) | — | exact | exact | bound or none |
| Energy score | = Euclidean error | samples | samples | samples |
| Variogram score (Scheuerer & Hamill, 2015) | — | samples | samples | samples |
| Interval coverage error | — | ✓ | ✓ | ✓ |

Gates: V2/V3 must pass V0's point gates plus macro CRPS and coverage error; V3
must also beat V2 on energy and variogram score. NLL is not a cross-version gate
because V3b has no exact likelihood.

---

## Open Points

- The noise-floor estimate rests on 658 repeated contact groups, half CircleG —
  treat as rough.
- The methods literature cited (Seitzer, Sohn, Shen & Meinshausen, Scheuerer &
  Hamill) is not in `papers/` and was not checked against the originals.
- Multimodality of sensor variation has not been tested.
- A replicate-residual diagnostics cell in `taxelnet/scripts/data_exploration.py`
  would make the statistics above reproducible; not yet added.
- Tests to write with V2/V3: β = 1 Laplace gradient equals `sign(mu - y)`;
  Laplace CRPS matches Monte Carlo; point-forecast CRPS = MAE and energy score =
  Euclidean error; `LowRankMultivariateNormal` NLL matches a dense multivariate
  normal.
