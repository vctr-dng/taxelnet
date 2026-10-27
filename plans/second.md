# Taxelnet: Audited Mesh-Deformation to Tactile-Map Pipeline

## Status and Purpose

This plan supersedes `plans/first.md` as the implementation plan. The first
plan established the right repository shape: a PyTorch library, dynamic grid
shapes, strict data handling, reusable training code, and tests. This revision
keeps those choices while replacing its provisional CSV assumptions with the
matched indenter dataset in `data/Indenters/`. Its schema is defined by
`data/Indenters/Dataset_Indenter_README.pdf` (2026-10-05). The facts below were
checked with `scripts/data_exploration.py` and `scripts/build_index.py`:

- Two row-aligned files, `Dataset_Indenter_Real.csv` and
  `Dataset_Indenter_Simulation.csv`, 12,730 rows each. Row i of one file and
  row i of the other are the same press.
- Each simulation row is one 77 x 49 height-field grid: the Z deformation in mm
  of the 3,773 top-layer pad nodes (Newton 1.5, QTAC100 model), stored as
  `n<node_id>` columns in row-major order.
- Each real row is one 7 x 4 tactile map, stored as `tax1 ... tax28` in
  row-major order.
- Both files start with the identical text columns `force, Path, dataset`.
  Several real presses can share one `(force, Path)` key. Each press keeps its
  own simulation row.
- On the current files, repeated simulation rows of a key are identical. The
  pipeline reports this property; it does not rely on it.
- `force` is stored with at most 4 decimals, so the keys match without
  trainer-side rounding. `dataset` (`01, 02, 03, 04, 06`) is constant within a
  key.

The legacy export in `data/raw/` (18-row long-form simulation blocks with
`data_*` taxel columns) is superseded. The new real data is that data with
`Flat/*`, CircleXS above 10 N, invalid simulations, and force ≤ 0 removed.

The goal is an auditable tactile surrogate:

```text
simulated mesh deformation grid -> predicted physical tactile taxel map
```

It must support credible comparison between training runs, deployment into
IsaacLab, and eventual modelling of physical measurement variation. V0 remains
a deterministic model.

## Non-Goals

- Do not retain TensorFlow, Keras, W&B, or the old hard-coded training script.
- Do not silently round force values, select arbitrary duplicate rows, or drop
  unmatched examples.
- Do not use a GNN: the input is a regular height-field lattice, not an
  irregular mesh-connectivity problem.
- Do not introduce probabilistic, VAE, or diffusion models before the
  deterministic baseline and data contract are validated.
- Do not make an RL policy part of this repository. Taxelnet produces tactile
  surrogate artifacts; IsaacLab consumes them.

## Data Contract

### Canonical Internal Representation

All readers produce grid records:

```text
GridTable
  keys:         [(force, Path), ...]      # one per CSV row; repeats allowed
  dataset:      [str, ...]                # provenance, not part of the key
  values:       (N, H, W) float32
  grid_shape:   (H, W)
  source_rows:  0-based CSV data-row index of each record
```

Every CSV row is one record. Records are never collapsed. The key is a raw
pair:

```text
(force, Path)
```

`force` must retain the parsed representation used in the CSV. The README
defines the match as "Path + force rounded to 4 decimals". The stored forces
already have at most 4 decimals, so Taxelnet applies no rounding. Instead, it
fails when any force has more than 4 decimals (`taxelnet.io_csv.rows_by_key`).

`Path` is `<Shape>/<Location>`, for example `TriangleS/Inferior`. Make the two
key-column names configurable, with `force,Path` as the defaults. Read
`dataset` as a string: it is a zero-padded id that pandas would parse as an
integer. Keep it as provenance and as a possible split unit.

### Matched Wide-Format Adapters

Both files share the leading columns `force, Path, dataset` with identical text.
Shared constants and loaders live in `taxelnet.io_csv`: `KEY`, `META_COLS`,
`SIM_GRID_SHAPE`, `TAXEL_GRID_SHAPE`, the pad extent, `read_matched_csv`,
`rows_by_key`, and `load_sim_values`.

#### Real tactile CSV

```text
force, Path, dataset, tax1 ... tax28
```

- `taxel_shape=(7, 4)`, row-major: `tax1..tax4` is the first row.
- Preserve every real row. Repeated rows for one key are repeated physical
  readings, not duplicate data to discard.
- Orientation is fixed by README §4. The real and simulated grids share their
  orientation: top-left is `tax1` / `n195`, at X = 41.80 mm, Y = −1.50 mm.
  Rows run along X, the pad length, and columns along Y, the width. No flip is
  applied. Record this orientation in the artifact contract.

#### Simulated deformation CSV

```text
force, Path, dataset, n195, n191, ..., n14899     # 3,773 node columns
```

- Node column k (0-based) is grid cell `r = k // 49`, `c = k % 49`, with
  `node_id = 3 + 4 * (48 - c) + 196 * r`. Assert that the header equals
  `taxelnet.io_csv.sim_node_columns()`, then reshape each row to 77 x 49.
- Reject the rotated `Extracted_simulation_data.csv` order (`n14899` first).
- Values: Z deformation in mm, 3 decimals, positive = pushed down, computed as
  Z(first frame) − Z(last HOLD frame). About 13% of values are small negatives
  (down to −0.103 mm), from the pad bulging next to the indenter. They are real
  signal.

### Pairing Rules

Pair one-to-one by row:

```text
real row i  <->  simulation row i
```

The `(force, Path)` key defines a **group** of presses for splitting and loss
balancing. Within a group, each real reading is paired with its own simulation
row. Repeated simulation rows are not assumed identical, and they are never
collapsed into one canonical grid.

`taxelnet.dataset.build_index` writes this pairing to `data/Indenters/index.json`,
one record per key:

```json
{"force": 40.0, "path": "TriangleM/Lateral2", "dataset": "02",
 "sim_id": [123, 124, ...], "real_id": [123, 124, ...]}
```

Each file is grouped on its own key columns. `build_index` requires
`sim_id == real_id` for every key.

Before building a dataset, the pairing code must fail with actionable reports
when any of these occur:

- The two files differ in row count, or their `force, Path, dataset` columns
  differ at any row.
- A key appears in only one file.
- A key spans several datasets.
- A force has more than 4 decimals.
- The simulation node columns are not in the expected order or count.
- A grid is incomplete, non-finite, or has an unexpected shape.
- A real target has non-finite values or an unexpected number of taxels.

The report should list a bounded sample of invalid keys and the total count.
Never silently inner-join and continue by default.

The number of keys whose simulation rows differ is reported in
`schema_report.json` as a diagnostic. It does not reject anything. It is 0 on
the current files.

### Excluded Data

The export leaves out 688 real presses: 601 CircleXS presses above 10 N (not
simulated), 83 with an invalid simulation, and 4 at force ≤ 0. The README
lists them in `Unmatched_real.txt`, but that file is not in `data/Indenters/`.
Obtain it, then copy the exclusion counts into `schema_report.json`. The
`Flat/*` paths of the legacy export are also absent.

## Split Design

### Partition Unit

A full raw `(force, Path)` group is indivisible. Every press of the group (real
reading and its simulation row) goes to exactly one partition:

```text
train OR validation OR test
```

This prevents the same simulated deformation condition from occurring in both
training and evaluation partitions.

### Train, Validation, and Test

> **Superseded (2026-10-14):** training data moved to `data/newton/v1/training/`, and the
> test set is now the external grasp pair in `data/newton/v1/test/` (see `data/README.md`).
> The training data is split into train/val only (`val_ratio`, default 10%); `test_ratio`,
> the immutable `splits/test_manifest.json`, and the in-distribution test carve-out are
> removed. The text below describes the previous design.

- There is one matched real-and-simulation pair and no separate test file.
  Carve train, validation, and test from it with one seeded, weighted
  group-level allocator. Group weight is the count of **real tactile
  readings**.
- Default targets are 80/10/10% of real readings (`val_ratio`, `test_ratio` in
  `DataConfig`), while preserving entire groups. Exact percentages are not
  always possible with uneven group sizes.
- Persist the test manifest once and treat it as immutable. Every later run
  reuses it, so that regression gates compare runs on the same test set.
- Validate that train, validation, and test key sets are pairwise disjoint.
- Persist `split_manifest.json` with every run: partition, raw key, dataset,
  number of real readings, source rows, seed, and configuration.

The default random group split evaluates interpolation to held-out conditions.
Stricter generalization tests are possible with `split_strategy=shape`, which
holds out whole `<Shape>`s, or `split_strategy=dataset`, which holds out a
whole `dataset` id. Dataset `01` is CircleXS only, and only 2 shapes appear in
more than one dataset, so a dataset split is effectively a split by shape
family. These are not V0 requirements.

### Group Weighting

One condition may have many real repeated measurements. During deterministic
training, aggregate losses within a group and average groups equally:

```text
group_loss = mean(loss(model(sim_grid_j), real_grid_j) for each press j in group)
batch_loss = mean(group_loss for each condition in the batch)
```

Apply the loss to every real reading before averaging. Never regress onto a
group-mean target: that changes the estimand (for L1, from the group median to
the group mean).

Implement either a group-aware batch sampler or a dataset/collate path that
returns all targets belonging to a condition. Do not let heavily repeated
conditions dominate merely because they have more recordings.

On `Dataset_Indenter_Real.csv`, raw-key grouping yields 9,237 groups over 219
paths. Of these, 8,691 (94%) are singletons; 546 repeated groups hold 4,039
readings (32%). The largest repeated groups are trials that ended at round
target forces (10, 20, 30, 50 N) for a few indenters; the largest is
CircleG/Corner4 at 10 N with 54 readings. CircleG supplies 53% of repeated
readings, DonutM 19%, and TriangleM 10%. Within one path, singletons are
effectively repeated trials at slightly different forces. Group balancing
therefore mostly down-weights those round-force trials, by up to 54x. Keep it as the default, and log the effective
weight share per path in `split_manifest.json`. A tempered group weight
`n_g ** alpha` (`alpha = 0` group-balanced, `alpha = 1` micro) is an optional
ablation, not the default.

Report both:

- **Micro metrics:** every real reading has equal weight.
- **Macro-group metrics:** every `(force, Path)` condition has equal weight.

## Preprocessing and Deployment Contract

### Training-Side Normalization

Do not hard-code the legacy positive-max normalization or runtime-only
clipping. Add a `PreprocessingConfig` with options such as:

```text
input_scaler:  identity | max_abs | zscore | robust_zscore
target_scaler: identity | per_taxel_zscore
clip_input:    optional [min, max]
```

- Fit every scaler on training groups only.
- Apply the fitted transform unchanged to validation, test, and inference.
- Save transform statistics, units, sign convention, grid orientation, and
  clipping configuration in `preprocessing.json`.
- Start V0 with an explicitly selected configuration after inspecting the
  dataset distribution; do not silently choose a scale from all data.
- Input units are mm of Z deformation, positive = pushed down, ranging from
  −0.103 to 2.319 mm on the current file. The small negative values are pad
  bulge, so `clip_input` must not clip at 0 by default.
- The target scaler changes the training loss: L1 on `per_taxel_zscore`
  targets equals raw L1 weighted by `1 / sigma_t` per taxel. Choose it
  deliberately. Compute every metric and regression gate in raw tactile counts
  (TC) after inverting the target transform. Likelihood metrics computed in
  scaled space convert to raw units by adding `sum_t log(sigma_t)`.

### IsaacLab Feature Contract

Taxelnet input CSVs are already height fields, generated with Newton 1.5 and
the QTAC100 pad model. Online IsaacLab inference starts with live pad node
positions, so define a reusable deployment contract that reproduces the
export:

```text
authored rest positions + live nodal positions
  -> rigid-motion alignment
  -> the 3,773 top-layer node IDs, in the order of sim_node_columns()
  -> Z(rest frame) - Z(current frame)      # + = pushed down, mm
  -> 77 x 49 deformation grid
  -> fitted Taxelnet normalization
  -> tactile prediction
```

The training data subtracts the first frame (pad at rest), which replaces a
separate no-contact baseline. The deformation is read at the last HOLD frame,
with the indenter at the target force.

The model bundle must include or reference:

```text
node_ids.csv                  # from the simulation column names, row-major 77 x 49
mesh grid shape               # (77, 49)
pad extent and node spacing   # X 1.80..41.80 mm, Y -1.50..-27.00 mm; 0.526 / 0.531 mm
taxel grid shape and orientation   # (7, 4), tax1 top-left, same as n195
simulator and pad model version    # Newton 1.5, QTAC100
rigid alignment method
rest-frame and HOLD-frame policy
input and target transforms
```

Do not claim deployment compatibility until replayed IsaacLab deformation grids
and training grids produce the same preprocessed tensor for a fixed fixture.

## Library Layout

`src/` remains reusable library code. Add or revise these modules:

```text
src/taxelnet/
  io_csv.py          # matched wide-format adapters (real tax*, sim n<id>), schema constants
  dataset.py         # build_index, strict row pairing, group-aware datasets
  splits.py          # seeded weighted group allocation and manifests
  preprocessing.py   # fit/apply transforms and tensor-contract metadata
  model.py           # deterministic V0 model and future model registry
  training.py        # group-balanced fit loop with MLflow logging hooks
  evaluation.py      # micro/macro metrics, predictions, calibration hooks
  checkpoint.py      # complete model-bundle save/load helpers
  export_onnx.py     # deterministic-model ONNX export and parity helpers
  config.py          # typed Data/Preprocessing/Model/Train/Eval configs
  artifacts.py       # dataset fingerprints, JSON metadata, model-bundle paths
```

Thin future entrypoints belong in `scripts/`:

```text
scripts/build_index.py        # done: writes <data-dir>/index.json
scripts/data_exploration.py   # done: marimo audit notebook for data/Indenters
scripts/train.py
scripts/evaluate.py
scripts/export_onnx.py
scripts/compare_runs.py
```

The utilities that came with the dataset (`new-dataset-utils/`) have been
integrated into these locations, and the directory has been removed:

- `build_index.py` became `taxelnet.dataset.build_index(real_csv, sim_csv)`,
  plus the thin CLI `scripts/build_index.py --data-dir data/Indenters`. Each
  record now lists every simulation row (`sim_id`), not only the first. Keys
  with differing simulation rows are counted instead of skipped.
- `explore_dataset.py` replaced the legacy `scripts/data_exploration.py`, as a
  marimo notebook. It imports its schema constants and loaders from
  `taxelnet.io_csv` and plots each real row under its own simulation row.

`data/Indenters/Plot_matched_dataset.py` is the dataset author's viewer. Keep
it untouched, as part of the data delivery.

## Training, Tracking, and Artifacts

### MLflow Is the System of Record

Use the existing `mlflow` dependency. Do not add W&B or make TensorBoard the
source of record.

Every training run logs:

```text
model architecture and parameter count
optimizer, loss, learning rate, batch size, seed, device
input/taxel shapes and orientation
preprocessing configuration and fitted statistics
dataset file checksums and schema report
split manifest and group-count summary
per-epoch train and validation micro/macro loss
best validation epoch and final test metrics
git revision and package versions
```

Use a configurable MLflow tracking URI. Local development may use a project
SQLite store; remote tracking must be supplied explicitly.

### Model Bundle

Every approved run produces a self-contained bundle:

```text
model.pt
best.pt
last.pt
preprocessing.json
node_ids.csv                 # when deployed from mesh nodes
schema_report.json
split_manifest.json
metrics.json
predictions_test.parquet
requirements.lock.txt
onnx/model.onnx              # after export
onnx/parity_report.json
```

`best.pt` must contain model state, model configuration, optimizer state,
epoch, best validation metric, MLflow run ID, and artifact-contract version.

### Regression Gates

`compare_runs.py` compares a candidate against an explicitly named approved
MLflow baseline on the same immutable test split. It returns non-zero when a
configured tolerance is exceeded.

V0 gates:

```text
test macro-group MAE
test micro MAE
test macro-group RMSE
per-taxel MAE
```

V2 and V3 candidates must also pass the V0 gates on their point estimate, plus:

```text
test macro-group CRPS
test interval coverage error (50, 80, 90, 95%)
test energy score             # V3 only
test variogram score          # V3 only
```

The comparison artifact includes candidate, baseline, absolute change, relative
change, tolerance, and pass/fail result. Do not compare runs trained against
different test manifests.

## Evaluation

### Deterministic V0 Metrics

At minimum compute:

```text
MAE
MSE
RMSE
per-taxel MAE and RMSE
micro metrics across all real readings
macro-group metrics across raw (force, Path) conditions
global SSIM with a fixed, documented data_range
```

Global SSIM is reported for comparison with the source papers, which evaluate
synthetic tactile maps by SSIM. It is not a training loss: windowed SSIM is
degenerate on 7 x 4 maps and unstable on near-zero maps.

Store two reference points in `metrics.json` next to the model MAE, both
computed on the evaluated partition:

- **Replicate noise floor:** MAE of contact readings against their repeated
  group's per-taxel median. On `Dataset_Indenter_Real.csv` it is 30.5 TC
  micro and 23.3 TC macro.
- **Trivial predictor:** MAE of predicting zero everywhere. On
  `Dataset_Indenter_Real.csv` it is 301 TC.

Write one prediction record per real target containing raw key, partition,
source-row provenance, flattened target, flattened prediction, and residual.
Persist the file with the model bundle and log it as an MLflow artifact.

### Probabilistic Metrics (V2 and Later)

Every model version is scored on one shared ladder so versions can be compared
directly. A deterministic prediction is a point-mass distribution: its CRPS
equals its MAE and its energy score equals its Euclidean error, so V0 remains a
valid baseline on the distributional metrics.

| Metric | V0 | V2 | V3a | V3b |
|---|---|---|---|---|
| MAE / RMSE of the point estimate (`mu` or sample median) | yes | yes | yes | yes |
| Per-taxel CRPS | equals MAE | closed form | samples | samples |
| NLL in raw units | none | exact | exact | bound or none |
| Energy score | Euclidean error | samples | samples | samples |
| Variogram score (Scheuerer & Hamill, 2015) | none | samples | samples | samples |
| Interval coverage error at 50, 80, 90, 95% | none | yes | yes | yes |

Report micro and macro-group values. Also report coverage and NLL separately on
held-out repeated groups, split into low-signal and high-signal taxels by
group-mean magnitude (the current data has no no-contact presses). NLL is not a
cross-version gate because V3b has no exact likelihood.

### Required Integrity Checks

- The model input/output shapes match the configured mesh/taxel shapes.
- All predictions and losses are finite.
- Test keys do not overlap train/validation keys unless explicitly allowed.
- Scalers were fitted from training groups only.
- The saved prediction file exactly covers the immutable test manifest.
- PyTorch and ONNX outputs agree on fixed test fixtures within an explicit
  tolerance after ONNX export.

## Implementation Sequence

1. Done: Add `data/README.md`, update `.gitignore` for raw data, checkpoints, MLflow
   outputs, and generated predictions. Document the `data/Indenters` schemas
   and required provenance, point to `Dataset_Indenter_README.pdf`, and mark
   `data/raw` as legacy.
2. Done: integrate `new-dataset-utils/` (`build_index`, schema constants,
   audit notebook) and remove the directory. Tests in `tests/test_index.py`.
3. Done: Implement the `io_csv.py` matched adapters (`GridTable` readers) with tests
   for valid rows, wrong node-column order or count, rotated export, missing
   columns, non-finite values, and forces with more than 4 decimals.
4. Done: Implement strict row pairing in `dataset.py` on top of `build_index`. Test
   row-wise key misalignment, unmatched keys, repeated real targets each paired
   with their own simulation row, and the differing-simulation-rows diagnostic.
   Cache the 77 x 49 simulation grids once as `.npy` or Parquet (about 192 MB
   in float32, against the 295 MB CSV), keyed by the CSV checksum.
5. Done: Implement `splits.py` weighted group allocation for train, validation, and
   test, with an immutable test manifest. Test that no group crosses
   partitions and target counts are as close as possible to requested
   proportions.
6. Done: Implement `preprocessing.py` and tests proving train-only fitted statistics
   are applied consistently to validation/test data.
7. Done: Complete `config.py`, `checkpoint.py`, and `artifacts.py` with typed,
   serializable configuration and bundle metadata.
8. Done: Complete deterministic V0 `model.py`, `training.py`, and `evaluation.py`.
   Use group-balanced mean absolute error as the primary loss. Log metrics and
   artifacts to MLflow.
9. Add CLI scripts for training, evaluation, MLflow baseline comparison, and
   ONNX export. Keep all business logic in `src/taxelnet/`. Done for
   `scripts/train.py`, which also evaluates the test split; the other three remain.
10. Implement ONNX export and parity tests. Validate the artifact with an ONNX
   Runtime replay fixture before integration with IsaacLab.
11. Run data exploration on the actual files, select the initial normalization
    configuration, train V0, designate an approved baseline, and activate the
    regression gate for subsequent candidates.

## Test and Verification Plan

Run before merging implementation work:

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyrefly check
pre-commit run --all-files
```

Required tests:

- CSV adapters and exact raw-key validation, including node-column order.
- `build_index`: `sim_id == real_id`, row misalignment, unmatched keys, and
  differing simulation rows (counted, not skipped).
- Strict one-to-one row pairing within `(force, Path)` groups.
- Weighted group splitting and manifest reproducibility.
- Normalization isolation from validation/test data.
- Deterministic model shape and finite-output checks.
- Two-epoch group-balanced training smoke test with MLflow local tracking.
- Evaluation micro/macro metric correctness.
- Checkpoint/model-bundle round trip.
- ONNX/PyTorch output parity.
- Regression comparison pass and deliberate-failure cases.

## Real-Sensor Variation Evidence

These statistics come from repeated real readings. The residual is each
reading minus its group mean, over groups with at least two readings. They are
the evidence that justifies the loss of each model version below, so recompute
them for every new dataset before relying on them.

The current file, `Dataset_Indenter_Real.csv` (computed 2026-10-07), has no
force = 0 presses: its smallest force is 0.42 N. All its repeated groups are
contact presses. The no-contact column below therefore comes from the legacy
`data/raw/Dataset_training_real.csv` (same sensor, computed 2026-10-02) and
cannot be recomputed on the current data.

| Quantity | No contact (legacy, force = 0) | Contact (current file) |
|---|---|---|
| Repeated groups / readings | 24 / 240 | 546 / 4,039 |
| Residual std | 11 TC | 69 TC |
| std / median absolute residual (Gaussian 1.48, Laplace 2.04) | 2.0 | 4.75 |
| Mean cross-taxel residual correlation | 0.05 | 0.47 |
| Residual variance in top two eigenmodes | 29% | 86% |
| Residual variance in a whole-map common offset | 7% | 41% |

Residual scale also grows with signal. On the current file, the median absolute
residual is about 10 TC where the group mean is below 10 TC, and it rises to
about 24 TC for group means of 580–1,140 TC before levelling off near 22 TC.
Taxels with group mean below 9 TC are 19% of the contact residuals. Their
median absolute residual is 10 TC, close to the legacy no-contact level, but
their std is 39 TC: the coherent contact offsets also reach the quiet taxels.
Real targets range from -888 to 2,711 TC, and 22% are negative after the
rest-baseline subtraction described in the dataset paper.

The legacy no-contact noise is small, independent across taxels, and roughly
Laplace. The contact variation is about 6x larger, heavier-tailed than Laplace,
and dominated by a coherent low-rank component, largely a whole-map offset.
This is consistent with the source paper's report of trial-to-trial hysteresis.
Pooling groups that have different noise scales inflates the std/median ratio,
so recheck tails on residuals scaled by V2's predicted `b`.

Because the current data has no no-contact presses, quantities derived from
the no-contact regime are configuration values with documented provenance.
This covers the Huber `delta` and V2's `b_floor`. Check each one on low-signal
taxels of the current data (median absolute residual about 10 TC).

## Model Architecture Roadmap

### V0: Deterministic Spatial CNN

V0 is intentionally deterministic. It predicts one tactile map for one
simulated deformation grid:

```text
x: (B, 1, H_mesh, W_mesh)          # (B, 1, 77, 49) on the current data
  -> Conv2d + ReLU + MaxPool blocks
  -> AdaptiveAvgPool2d(H_taxel, W_taxel)
  -> 1 x 1 convolution
  -> y_hat: (B, H_taxel, W_taxel)
```

Use the existing `CNNTaxelPredictor` as the starting point, with a linear final
head because baseline-corrected physical taxel values may be negative. The
77 x 49 input has about 16x more cells than the legacy 18 x 12 grid, so use
3–4 downsampling blocks (77 x 49 -> about 9 x 6) before
`AdaptiveAvgPool2d(7, 4)`. The aspect ratios are close (77/49 ≈ 1.57 against
7/4 = 1.75), and both grids share the same orientation, so adaptive pooling
maps pad regions onto taxels consistently. Train it
with group-balanced mean absolute error. This is the primary V0 artifact and
the first regression baseline.

```text
L_V0 = mean over groups g [ mean over presses j in g [ mean over taxels t
         |mu(x_j)_t - y_{j,t}| ] ]
```

- L1 estimates the conditional per-taxel median. The heavy-tailed contact
  residuals rule out MSE as the training loss; MSE and RMSE remain evaluation
  metrics.
- Any point loss blurs peaks where contact position varies between trials.
  That is a limit of point prediction, not a defect to tune away, and it is
  part of the motivation for V3.
- The L1 gradient does not shrink near the optimum, so use a learning-rate
  schedule (cosine or reduce-on-plateau).
- Huber loss with `delta` near the low-signal noise level (10-20 TC: legacy
  no-contact std 11 TC, current low-signal median absolute residual 10 TC) is an
  ablation only. Its estimand lies between the median and the mean.

The old 18 x 12 -> 28 dense-head architecture matches no current data, so it
is not carried forward. A `LegacyDenseHeadCNN` reproduction is only meaningful
against `data/raw`. Summed L1 is 28 x MAE. Adam is approximately invariant to
that constant, so the two losses train alike and differ only in reported
magnitude.

### V1: Deterministic U-Net-Style Decoder

If V0 underfits spatial detail, replace adaptive pooling with a compact decoder
and encoder skip connections. It remains deterministic and keeps the same
input/output contract, manifests, MLflow metrics, and ONNX parity requirement.

### V2: Mean Plus Aleatoric Uncertainty

Only after deterministic V0 is approved, add two spatial heads:

```text
mu(x):    conditional per-taxel median tactile map (Laplace location)
b(x):     positive per-taxel physical-variation scale (Laplace scale)
```

Under a Laplace likelihood `mu` is the conditional median, not the expected
map. Warm-start `mu` from the approved V0 checkpoint.

#### Loss

Train with the beta = 1 variant of the heteroscedastic Laplace negative
log-likelihood (Seitzer et al., ICLR 2022), with the same group balancing as
V0:

```text
l(y, mu, b) = sg(b) * ( |y - mu| / b + log(b) )     # sg = stop-gradient
b           = b_min + softplus(raw_scale(x))
```

- With beta = 1, the gradient on `mu` is exactly `sign(mu - y)`, so `mu`
  trains with the same objective as V0 and V0's gates still apply to it.
- The gradient on `b` is `1 - |y - mu| / b`, so `b` converges to the expected
  absolute residual, which is the Laplace maximum-likelihood scale.
- The plain heteroscedastic NLL divides the `mu` gradient by `b`. That starves
  high-variation regions. In the current data the contact residual scale
  varies about 2.4x with signal, and it is about 6x the legacy no-contact
  noise.
- Set `b_min` to a few tactile counts so the NLL cannot diverge on
  near-constant taxels.

#### Scale Parametrization

Repeated real measurements for one condition provide the direct evidence for
aleatoric variation. Singleton groups do help train `b(x)`, because its
parameters are shared across inputs. The risk is that `b` also absorbs model
misfit and sim-to-real bias, and training residuals are optimistic. Build the
scale on a physical baseline fit from repeated groups:

```text
b0(x, taxel) = b_floor(taxel) + kappa(taxel) * g(|mu(x, taxel)|)
b(x, taxel)  = b0(x, taxel) * exp(delta(x, taxel))
```

- `b_floor` is the low-signal noise level. It is a configuration value: the
  legacy no-contact std is 11 TC, and on the current file, low-signal taxels
  have a median absolute residual of 10 TC, which is a Laplace scale of about
  15 TC.
- `g` is a saturating function, so the scale grows with signal and then levels
  off as observed.
- Shrink `delta` toward zero with a strong penalty.
- Do not use a single global per-taxel scale as the baseline. Covering the
  2.4x signal-dependent range in the current data needs `|delta|` of about 0.9.
  Covering the 6x legacy no-contact to contact gap needs about 1.8. Strong
  shrinkage would prevent either, miscalibrating both ends.
- CircleG supplies 53% of repeated readings, so treat the fitted baseline as a
  prior, not the truth.

After training, fit a per-taxel multiplier on `b` using the validation split,
so that interval coverage is calibrated on held-out conditions. Report coverage
on the test split.

#### Tail Check

Scaled residuals `|y - mu| / b` should follow Exp(1). If they remain
heavy-tailed, which the contact std/median ratio of 4.9 suggests, replace the
Laplace likelihood with a Student-t likelihood (`torch.distributions.StudentT`)
with a learned global degrees-of-freedom parameter. Keep the beta = 1 weighting
so `mu` still trains as L1.

#### Evaluation

Evaluate with the probabilistic metric ladder, in addition to V0 point metrics:

- **Closed-form Laplace CRPS:**
  `|y - mu| + b * exp(-|y - mu| / b) - 3 * b / 4`.
- **Central interval for level p:** `mu +/- b * ln(1 / (1 - p))`.
- **Diagnostics:** a PIT histogram, plus NLL and coverage on held-out repeated
  groups, split into low-signal and high-signal taxels.

V2 samples are independent across taxels. With a mean contact residual
correlation of 0.47, sampled maps show incoherent per-taxel noise rather than
the coherent shifts the sensor produces. Export `mu` and intervals for
IsaacLab; do not use V2 samples for tactile domain randomization.

### V3: Correlated Conditional Tactile Samples

If independent per-taxel uncertainty cannot represent coherent sensor changes,
use a conditional latent residual model:

```text
deformation grid + latent z -> plausible full tactile map
```

This is appropriate only when repeated measurements are sufficiently numerous
and residual analysis demonstrates spatially correlated or multimodal sensor
variation. The current contact residuals already show spatial correlation:
mean correlation 0.47, 84% of variance in two eigenmodes, and 41% in a
whole-map offset. Multimodality has not been tested. V3 must be compared against
V2 using the probabilistic metric ladder, and must beat V2 on energy score and
variogram score without losing CRPS or point MAE. It must never replace the
deterministic V0 mean baseline without retaining it.

Keep `mu` trained by L1 in every V3 variant, frozen from V0/V2 or decoupled by
stop-gradient, so the distributional loss cannot shift the point estimate.

#### V3a: Low-Rank Plus Diagonal Gaussian (First)

```text
y | x ~ Normal(mu(x), W(x) W(x)^T + diag(d(x))),   rank K = 2-4
W(x) = B * diag(a(x))     # B: fixed 28 x K basis from PCA of replicate residuals
```

- Train the covariance by exact group-balanced NLL with
  `torch.distributions.LowRankMultivariateNormal`.
- A fixed residual basis `B` (common offset and gain modes) keeps the model
  data-efficient; only the per-input amplitudes `a(x)` and `d(x)` are learned.
- This is a linear-decoder conditional VAE with an exact likelihood. It has no
  ELBO gap and cannot suffer posterior collapse.
- Sampling is `mu + W z + sqrt(d) * eps`. It exports to ONNX with `z` and
  `eps` as inputs.

#### V3b: Nonlinear Generator (Only If Needed)

Use V3b only if V3a loses on joint scores, or if residual analysis shows
multimodal variation.

Preferred: an energy-score generator (engression; Shen & Meinshausen), trained
with `m >= 2` noise draws per input:

```text
y_hat_j = g(x, eps_j)
ES      = mean_j ||y_hat_j - y||
          - 1 / (2 m (m - 1)) * sum_{j != k} ||y_hat_j - y_hat_k||
```

- The energy score is a proper scoring rule, so the training loss is also the
  comparison metric.
- There is no encoder, so systematic misfit cannot leak into a latent code, and
  there is no posterior to collapse.
- No likelihood family has to be chosen.
- It deploys as ONNX with a noise input.
- For repeated groups, average the score over every real reading in the group,
  then balance across groups.
- The energy score is weakly sensitive to correlation structure, so add a
  variogram-score term.

Alternative: a conditional VAE (Sohn et al., NeurIPS 2015) trained on

```text
-E_q[log p(y | x, z)] + beta * KL(q(z | x, y) || p(z | x))
```

with these safeguards:

- Use a Laplace decoder likelihood with V2's learned scale. A fixed-scale L1
  reconstruction is an implicit, untuned beta.
- Use free bits or KL annealing, and monitor KL per latent dimension. `mu(x)`
  already explains most variance, so posterior collapse is likely.
- Check whether the encoder stores sim-to-real misfit in `z`. Prior samples
  would then inject systematic bias that looks like sensor noise.
- Use Sohn et al.'s hybrid objective for the mismatch between training on
  posterior `z` and deploying with prior `z`.
- Its NLL is only available as an importance-weighted bound, so do not use NLL
  to compare it with V2 or V3a.
