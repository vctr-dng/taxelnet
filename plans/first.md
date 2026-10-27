# Taxelnet: mesh-deformation → tactile-map training pipeline

## Context

`taxelnet` is a brand-new repo (uv-managed, Python 3.13, `torch`/`torchvision`/`numpy`/`tqdm`/`marimo` already in `pyproject.toml`, `src/taxelnet/__init__.py` empty, pre-commit wired for ruff + pyrefly type checking). The goal is a neural network that predicts a tactile sensor reading (an array of taxel values) from a soft-body simulation's mesh deformation, trained against real experimental tactile data.

Clarified with the user:
- **Input**: mesh vertices sit on a **fixed, regularly-spaced x,y lattice** (rest-pose positions constant across samples); the only per-sample varying field is the **z (height-field) displacement**. So the mesh deformation map is a single-channel 2D grid — structurally the same kind of object as the taxel output, just at a different (much finer) resolution.
- **Output**: a **regular 2D taxel grid** (H×W scalar values), coarser than the mesh grid.
- **Pairing**: mesh data and taxel data live in **separate CSV files**, joined on two columns: `object_name` and `force`.
- **Data status**: collected but not yet on disk — the user will drop files in manually later. Exact file names, column layout, and scale (mesh grid size, taxel grid size, sample count) are **unknown now**, so the pipeline must infer shapes at runtime and document its schema assumptions clearly rather than hardcoding guesses.
- **Splits**: the user will provide **one file pair for train/val** and **a separate file pair for test** — no object/force-based splitting needed; the pipeline only needs to carve a val subset out of the train/val pair (default: random 90/10 by sample index, configurable).

Because the input is a regular grid (not an arbitrary point set or unknown graph), neither an MLP-on-flattened-vector nor a GNN is the right fit: a GNN earns its keep on irregular/unknown connectivity, which doesn't apply to a lattice, and a flatten+MLP would discard the 2D locality that's directly usable here. The V1 model is instead a **CNN encoder** that downsamples the fine mesh grid to the coarser taxel grid resolution, since convolutions exploit 2D spatial locality directly and need far fewer parameters than a dense layer for this kind of resolution-changing grid regression. No new dependency is needed for this (`torch.nn` already covers it).

## Directory layout

`src/` holds only reusable, importable, testable library code. Training/evaluation entrypoints live in `scripts/` instead of inside the package — kept as a placeholder here; the user will decide later (in a separate plan) whether each is a plain Python script or a marimo notebook, and build it then. Nothing under `scripts/` is implemented by this plan.

```
taxelnet/
├── pyproject.toml              # + pytest as dev dependency
├── .gitignore                  # + /data/*, !/data/README.md, /checkpoints/, /runs/
├── data/
│   └── README.md               # documents expected file names + CSV schema (assumption, flagged)
├── src/taxelnet/
│   ├── __init__.py
│   ├── config.py                # DataConfig / ModelConfig / TrainConfig dataclasses + argparse
│   ├── io_csv.py                 # schema-aware CSV → numpy array readers
│   ├── dataset.py                # join on (object_name, force) + torch Dataset
│   ├── model.py                   # CNNTaxelPredictor baseline
│   ├── checkpoint.py              # save/load helpers
│   ├── training.py                # reusable train_one_epoch / evaluate_loss / fit() loop
│   └── evaluation.py              # reusable evaluate_test_set() metrics computation
├── scripts/                       # NOT built by this plan — placeholder only
│   └── (train/evaluate entrypoint, format TBD in a future plan)
└── tests/
    ├── conftest.py                # synthetic CSV fixtures
    ├── test_io_csv.py
    ├── test_dataset.py
    ├── test_model.py
    └── test_training_smoke.py     # tiny 2-epoch run calling taxelnet.training.fit() directly
```

This split matters because `train.py`/`evaluate.py` as CLI entrypoints mix two concerns: the actual training-loop logic (reusable, unit-testable, importable from a script, a notebook, or a test) and the glue that wires up argument parsing / notebook UI around it (entrypoint-specific, not meant to be imported elsewhere). Putting the loop itself in `src/taxelnet/training.py` and `evaluation.py` means `scripts/` — whatever shape it ends up taking — stays a thin wrapper that just calls into `src/`, and the tests exercise the reusable logic directly without depending on a CLI or notebook runtime.

## Assumed CSV schema (explicitly flagged — confirm once real files arrive)

Both files now share the **same structural schema** — one row per `(object_name, force)` trial, with a regular grid of scalar values — just at different resolutions (mesh grid is much finer than the taxel grid):
```
object_name, force, r0_c0, r0_c1, ..., r{H-1}_c{W-1}
```
Grid shape `(H, W)` is inferred from `r{row}_c{col}`-style column names (loader validates the full H×W set is present, no gaps). If real columns don't encode row/col this way, an explicit `grid_shape: tuple[int,int] | None` override is accepted per-file, with all non-key columns then treated as a flat row-major grid. Because both files share this schema, a single `read_grid_csv(path, grid_shape=None) -> GridTable` function handles both — no separate mesh/taxel parsing logic.

x,y vertex positions are not in the CSV at all since they're fixed/regular — only the per-cell z-displacement (mesh) or per-cell reading (taxel) varies per sample, consistent with the "z-only height field" input.

**Join key**: `(object_name, force)`. `force` is read and compared **as a string** (not float) to avoid float round-tripping bugs between independently-generated sim/real exports; a parsed float `force` is also exposed as a feature. Join is an inner join, `strict=True` by default — raises listing unmatched keys rather than silently dropping rows (set `strict=False` to allow silent drop with a warning).

No input/output normalization in V1 (avoids hardcoding an unknown data scale) — flagged as a TODO once real data distribution is known (e.g. train-split-only z-scoring).

## Core modules

- **`io_csv.py`**: `read_grid_csv(path, grid_shape=None) -> GridTable` — one shared reader for both mesh and taxel files (frozen dataclass holding parsed `(object_name, force)` keys + a `(num_rows, H, W)` numpy array + inferred grid shape). Pure parsing/validation, no joining.
- **`dataset.py`**: `join_tables(mesh, taxel, strict=True) -> list[JoinedSample]`; `TaxelDataset(torch.utils.data.Dataset)` with `.from_csv(mesh_path, taxel_path, mesh_grid_shape=None, taxel_grid_shape=None, strict=True)` classmethod, `__getitem__` returning `(mesh_grid: (1, H_mesh, W_mesh), taxel_grid: (H_taxel, W_taxel))` tensors (channel dim added for conv input), and `mesh_shape` / `taxel_shape` properties used downstream to size the model (never hardcoded). `random_split_dataset(dataset, val_ratio=0.1, seed=0)` for the train/val carve-out.
- **`model.py`**: `CNNTaxelPredictor(taxel_shape, base_channels=16, num_down_blocks=3)` — a stack of `Conv2d + ReLU + MaxPool2d` blocks progressively downsampling the fine mesh grid, independent of the exact mesh/taxel resolution ratio, followed by `nn.AdaptiveAvgPool2d(taxel_shape)` to force the exact target spatial size regardless of how the ratio divides, then a `Conv2d(..., 1, kernel_size=1)` to collapse to a single output channel and `squeeze(1)` to `(B, H_taxel, W_taxel)`. Using adaptive pooling as the resolution bridge means the model doesn't need to know the mesh/taxel resolution ratio in advance — it works whether the ratio is exactly known or only roughly characterized as "mesh is much finer." Docstring notes that a fuller U-Net-style decoder (skip connections, learned upsampling) is a reasonable future upgrade if the plain downsampling CNN underfits, but isn't built now.
- **`config.py`**: `DataConfig` (CSV paths, optional `mesh_grid_shape`/`taxel_grid_shape` fallbacks, `val_ratio`, `split_seed`, `strict_join`), `ModelConfig` (`base_channels`, `num_down_blocks`), `TrainConfig` (epochs, batch size, lr, device, checkpoint/log dirs, seed), plus `build_arg_parser()` / `parse_configs(argv)` for CLI overrides. No YAML/Hydra.
- **`checkpoint.py`**: `save_checkpoint(path, model, optimizer, epoch, extra=None)`, `load_checkpoint(path, model, optimizer=None, map_location="cpu")`.
- **`training.py`**: `train_one_epoch(model, loader, optimizer, loss_fn, device) -> float`, `evaluate_loss(model, loader, loss_fn, device) -> float`, and `fit(model, train_loader, val_loader, train_cfg) -> TrainingHistory` — orchestrates the full epoch loop, `torch.utils.tensorboard.SummaryWriter` logging (ships with torch — no new dependency; viewing logs later needs `tensorboard` installed separately, not a project dependency), `tqdm` progress, and checkpointing (`last.pt` every epoch, `best.pt` on val-loss improvement) via `checkpoint.py`. This is the function a future `scripts/train.py` (or a marimo notebook) calls directly — it takes an already-built model/loaders/config rather than parsing CLI args itself, so it's equally usable from a script, a notebook cell, or a test.
- **`evaluation.py`**: `evaluate_test_set(model, test_loader, device) -> dict[str, float]` — runs a no-grad forward pass over the test set, returns aggregate MSE/MAE (and optionally per-sample arrays for a caller that wants to dump predictions). Same design intent: callable from any future entrypoint without needing a CLI.

## New dependency

Add `pytest` as a **dev dependency** (`uv add --dev pytest`) — needed to test the CSV-join and shape-inference logic against synthetic fixtures before real data exists. No other new dependencies.

## Verification

Since `scripts/` isn't built by this plan, verification stays at the library level: everything in `src/taxelnet/` must be independently correct and usable by whatever entrypoint comes later.

1. `uv add --dev pytest`, then `uv run pytest` — covers CSV parsing/validation, the `(object_name, force)` join (including the mismatched-key error path), dataset tensor shapes/dtypes, model forward-pass shape, and a 2-epoch smoke run calling `taxelnet.training.fit(...)` directly on synthetic data (asserts no exception, checkpoint file written, loss is finite) plus `taxelnet.evaluation.evaluate_test_set(...)` on a synthetic test set.
2. `uv run ruff check`, `uv run ruff format --check`, `uv run pyrefly check`, then `pre-commit run --all-files` as the final gate.
3. End-to-end exercise via an actual `scripts/` entrypoint (plain script or marimo notebook) is explicitly deferred to the follow-up plan that builds `scripts/`.

## Critical files to create

- `src/taxelnet/io_csv.py`
- `src/taxelnet/dataset.py`
- `src/taxelnet/model.py`
- `src/taxelnet/config.py`
- `src/taxelnet/checkpoint.py`
- `src/taxelnet/training.py`
- `src/taxelnet/evaluation.py`
- `data/README.md`
- `tests/conftest.py`, `tests/test_io_csv.py`, `tests/test_dataset.py`, `tests/test_model.py`, `tests/test_training_smoke.py`
- `.gitignore`, `pyproject.toml` (dev dep addition)

Not built by this plan: `scripts/` entrypoints (format — plain script vs. marimo notebook — to be decided in a separate follow-up plan).
