"""Schema-aware CSV -> GridTable parsing for the matched indenter files.

The matched indenter export (`data/Indenters`, see Dataset_Indenter_README.pdf) is two
row-aligned files: row i of the real file and row i of the simulation file are the same
press. Both start with the text columns `force, Path, dataset`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

REAL_CSV = "Dataset_Indenter_Real.csv"
SIM_CSV = "Dataset_Indenter_Simulation.csv"

KEY = ["force", "Path"]  # matching key of a press
DATASET_COL = "dataset"  # zero-padded real dataset id ("01", ...): provenance, not part of the key
META_COLS = [*KEY, DATASET_COL]

# grids, row-major as stored (see data/Indenters/Plot_matched_dataset.py)
SIM_GRID_SHAPE = (77, 49)  # simulated nodes: rows along X, columns along Y
TAXEL_GRID_SHAPE = (7, 4)  # real taxels: tax1..tax4 is the first row
X_MIN, X_MAX, Y_MIN, Y_MAX = 1.798, 41.802, -27.001, -1.499  # node rest positions (mm)

MAX_REPORTED = 5  # sample size of invalid rows/keys listed in error messages


def sim_node_columns() -> list[str]:
    """Expected simulation columns in stored order: n195 top-left ... n14899 bottom-right."""
    n_rows, n_cols = SIM_GRID_SHAPE
    return [
        f"n{3 + 4 * (n_cols - 1 - c) + 196 * r}"
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def taxel_columns() -> list[str]:
    return [f"tax{k}" for k in range(1, TAXEL_GRID_SHAPE[0] * TAXEL_GRID_SHAPE[1] + 1)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_matched_csv(csv: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read a matched real or simulation CSV, keeping `dataset` as a string."""
    return pl.read_csv(csv, columns=columns, schema_overrides={DATASET_COL: pl.String})


def check_force_decimals(df: pl.DataFrame, name: str, force_col: str = "force") -> None:
    off_grid = df.filter(pl.col(force_col).round(4) != pl.col(force_col))
    if not off_grid.is_empty():
        raise ValueError(
            f"{name}: {off_grid.height} forces have more than 4 decimals, "
            f"e.g. {off_grid[force_col].head(MAX_REPORTED).to_list()}"
        )


def rows_by_key(csv: Path, key_cols: list[str] = KEY) -> pl.DataFrame:
    """0-based data-row indices (`id`) and datasets per key pair, reading only the key columns."""
    df = read_matched_csv(csv, columns=[*key_cols, DATASET_COL])
    check_force_decimals(df, csv.name, key_cols[0])
    return (
        df.with_row_index("id")
        .group_by(key_cols, maintain_order=True)
        .agg(pl.col(DATASET_COL).unique(), pl.col("id"))
    )


def _check_columns(name: str, found: list[str], expected: list[str], what: str) -> None:
    if found != expected:
        missing = [c for c in expected if c not in set(found)]
        extra = [c for c in found if c not in set(expected)]
        raise ValueError(
            f"{name}: {what} columns differ from the expected {len(expected)} "
            f"({expected[0]} ... {expected[-1]}): found {len(found)}, "
            f"{len(missing)} missing (e.g. {missing[:MAX_REPORTED]}), "
            f"{len(extra)} unexpected (e.g. {extra[:MAX_REPORTED]})"
            + ("; same columns in another order" if not missing and not extra else "")
        )


def _check_finite(name: str, values: np.ndarray) -> None:
    bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if bad.size:
        raise ValueError(
            f"{name}: {bad.size} rows have missing or non-finite values, "
            f"e.g. data rows {bad[:MAX_REPORTED].tolist()}"
        )


def load_sim_values(csv: Path) -> np.ndarray:
    """Node values of every simulation row, shape (n_rows, n_nodes), in stored node order."""
    df = read_matched_csv(csv)
    node_cols = [c for c in df.columns if c not in META_COLS]
    if node_cols != sim_node_columns():
        raise ValueError(
            f"{csv.name}: node columns are not n195 ... n14899 in row-major order "
            f"(found {node_cols[0]} ... {node_cols[-1]}, {len(node_cols)} columns); "
            "a rotated Extracted_simulation_data.csv export is not supported"
        )
    return df.select(node_cols).cast(pl.Float32).to_numpy()


@dataclass(frozen=True)
class GridTable:
    """One record per CSV data row; records are never collapsed."""

    keys: list[tuple[float, str]]  # raw (force, Path) per row; repeats allowed
    dataset: list[str]  # provenance, not part of the key
    values: np.ndarray  # (N, H, W) float32
    grid_shape: tuple[int, int]
    source_rows: np.ndarray  # 0-based CSV data-row index of each record


def _meta(csv: Path, key_cols: list[str]) -> tuple[list[tuple[float, str]], list[str]]:
    df = read_matched_csv(csv, columns=[*key_cols, DATASET_COL])
    check_force_decimals(df, csv.name, key_cols[0])
    keys = list(
        zip(df[key_cols[0]].cast(pl.Float64).to_list(), df[key_cols[1]].to_list())
    )
    return keys, df[DATASET_COL].to_list()


def read_real_table(csv: Path, key_cols: list[str] = KEY) -> GridTable:
    """Real tactile file: `force, Path, dataset, tax1 ... tax28` -> (N, 7, 4), no flip."""
    df = read_matched_csv(csv)
    _check_columns(
        csv.name, [c for c in df.columns if c not in [*key_cols, DATASET_COL]],
        taxel_columns(), "taxel",
    )  # fmt: skip
    values = df.select(taxel_columns()).cast(pl.Float32).to_numpy()
    _check_finite(csv.name, values)
    keys, dataset = _meta(csv, key_cols)
    return GridTable(
        keys=keys,
        dataset=dataset,
        values=values.reshape(-1, *TAXEL_GRID_SHAPE),
        grid_shape=TAXEL_GRID_SHAPE,
        source_rows=np.arange(len(keys)),
    )


def read_sim_table(
    csv: Path, key_cols: list[str] = KEY, cache_dir: Path | None = None
) -> GridTable:
    """Simulation file: `force, Path, dataset, n195 ... n14899` -> (N, 77, 49).

    With `cache_dir`, the parsed node values are cached as `sim_<sha256[:16]>.npy`, keyed
    by the CSV checksum, so the 295 MB CSV is parsed once.
    """
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f"sim_{file_sha256(csv)[:16]}.npy"
    if cache is not None and cache.exists():
        values = np.load(cache)
    else:
        values = load_sim_values(csv)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, values)
    n_nodes = SIM_GRID_SHAPE[0] * SIM_GRID_SHAPE[1]
    if values.ndim != 2 or values.shape[1] != n_nodes:
        raise ValueError(
            f"{csv.name}: expected {n_nodes} nodes per row, got {values.shape}"
        )
    _check_finite(csv.name, values)
    keys, dataset = _meta(csv, key_cols)
    if len(keys) != values.shape[0]:
        raise ValueError(
            f"{csv.name}: {len(keys)} rows but {values.shape[0]} grids (stale cache?)"
        )
    return GridTable(
        keys=keys,
        dataset=dataset,
        values=values.reshape(-1, *SIM_GRID_SHAPE),
        grid_shape=SIM_GRID_SHAPE,
        source_rows=np.arange(len(keys)),
    )
