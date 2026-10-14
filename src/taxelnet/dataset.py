"""Strict row pairing of the matched real and simulation files + group-aware torch Dataset."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from taxelnet.config import DataConfig
from taxelnet.io_csv import (
    DATASET_COL,
    KEY,
    MAX_REPORTED,
    load_sim_values,
    read_real_table,
    read_sim_table,
    rows_by_key,
)


def build_index(
    real_csv: Path,
    sim_csv: Path,
    sim_values: np.ndarray | None = None,
    key_cols: list[str] = KEY,
) -> tuple[list[dict], int]:
    """One record per (force, Path) key pair, and the number of key pairs whose sim rows differ.

    `path` holds the second key column (`Path` for training files, `Name` for test files);
    `dataset` is "" when the files have no `dataset` column.

        {"force": 40.0, "path": "TriangleM/Lateral2", "dataset": "02",
         "sim_id": [123, 124, ...], "real_id": [123, 124, ...]}

    `sim_id` / `real_id` are 0-based data-row indices (header excluded) into the simulation /
    real CSV. Each file is grouped on its own key columns, and the two id lists must be equal:
    the files are row-aligned, so real row i pairs with sim row i. Repeated sim rows of a key
    pair are not assumed identical; differing ones are only counted. `sim_values` (n_rows,
    n_nodes) avoids parsing the simulation CSV again when it is already loaded.
    """
    real = rows_by_key(real_csv, key_cols)
    sim = rows_by_key(sim_csv, key_cols)

    joined = real.join(
        sim, on=key_cols, how="full", validate="1:1", suffix="_sim", coalesce=True
    )
    unmatched = joined.filter(pl.col("id").is_null() | pl.col("id_sim").is_null())
    if not unmatched.is_empty():
        raise ValueError(
            f"{unmatched.height} key pairs are not in both files, "
            f"e.g. {unmatched.select(key_cols).head(MAX_REPORTED).rows()}"
        )
    misaligned = joined.filter(
        (pl.col("id") != pl.col("id_sim"))
        | (pl.col(DATASET_COL) != pl.col(f"{DATASET_COL}_sim"))
    )
    if not misaligned.is_empty():
        raise ValueError(
            f"{misaligned.height} key pairs have different rows or datasets in the "
            f"two files, e.g. {misaligned.select(key_cols).head(MAX_REPORTED).rows()}"
        )
    multi = joined.filter(pl.col(DATASET_COL).list.len() != 1)
    if not multi.is_empty():
        raise ValueError(
            f"{multi.height} key pairs span several datasets, "
            f"e.g. {multi.select(key_cols).head(MAX_REPORTED).rows()}"
        )

    if sim_values is None:
        sim_values = load_sim_values(sim_csv, key_cols)
    sim_values = sim_values.reshape(sim_values.shape[0], -1)
    index = []
    n_sim_differ = 0
    for row in joined.sort(key_cols).iter_rows(named=True):
        sim_rows = sim_values[row["id_sim"]]
        n_sim_differ += not (sim_rows == sim_rows[0]).all()
        index.append(
            {
                "force": row[key_cols[0]],
                "path": row[key_cols[1]],
                "dataset": row[DATASET_COL][0],
                "sim_id": row["id_sim"],
                "real_id": row["id"],
            }
        )
    return index, n_sim_differ


@dataclass(frozen=True)
class PairedData:
    """Every real reading paired with its own simulation row; rows are never collapsed."""

    x: np.ndarray  # (N, H_mesh, W_mesh) float32 simulated deformation (mm)
    y: np.ndarray  # (N, H_taxel, W_taxel) float32 real tactile map (TC)
    group: np.ndarray  # (N,) int64 index into `groups`
    source_rows: np.ndarray  # (N,) 0-based CSV data row (identical in both files)
    groups: list[dict]  # build_index records: force, path, dataset, sim_id, real_id
    files: tuple[Path, Path]  # (real_csv, sim_csv)
    n_sim_differ: int

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self.x.shape[1], self.x.shape[2]

    @property
    def taxel_shape(self) -> tuple[int, int]:
        return self.y.shape[1], self.y.shape[2]

    @property
    def group_sizes(self) -> np.ndarray:
        return np.bincount(self.group, minlength=len(self.groups))


def load_paired_files(
    real_csv: Path,
    sim_csv: Path,
    key_cols: list[str] = KEY,
    cache_dir: Path | None = None,
) -> PairedData:
    """Read, validate and pair both files; fails with a bounded report on any violation."""
    real = read_real_table(real_csv, key_cols)
    sim = read_sim_table(sim_csv, key_cols, cache_dir)

    if len(real.keys) != len(sim.keys):
        raise ValueError(
            f"{real_csv.name} has {len(real.keys)} rows but {sim_csv.name} has {len(sim.keys)}"
        )
    differ = [
        i
        for i, (rk, sk, rd, sd) in enumerate(
            zip(real.keys, sim.keys, real.dataset, sim.dataset)
        )
        if rk != sk or rd != sd
    ]
    if differ:
        raise ValueError(
            f"{len(differ)} rows have different {', '.join([*key_cols, DATASET_COL])} "
            f"in the two files, e.g. data rows {differ[:MAX_REPORTED]}"
        )

    groups, n_sim_differ = build_index(real_csv, sim_csv, sim.values, key_cols)
    group = np.full(len(real.keys), -1, dtype=np.int64)
    for g, record in enumerate(groups):
        group[record["real_id"]] = g
    assert (group >= 0).all(), "build_index must cover every row"

    return PairedData(
        x=sim.values,
        y=real.values,
        group=group,
        source_rows=real.source_rows,
        groups=groups,
        n_sim_differ=n_sim_differ,
        files=(real_csv, sim_csv),
    )


def load_paired(data_cfg: DataConfig) -> PairedData:
    """The training pair (`data_dir`), split later into train and val."""
    return load_paired_files(
        data_cfg.data_dir / data_cfg.real_csv,
        data_cfg.data_dir / data_cfg.sim_csv,
        list(data_cfg.key_cols),
        data_cfg.cache_dir,
    )


def load_test_paired(data_cfg: DataConfig) -> PairedData:
    """The external test pair (`test_dir`), never used for fitting or model selection."""
    return load_paired_files(
        data_cfg.test_dir / data_cfg.test_real_csv,
        data_cfg.test_dir / data_cfg.test_sim_csv,
        list(data_cfg.test_key_cols),
        data_cfg.cache_dir,
    )


class TaxelDataset(Dataset):
    """Rows of a partition with fitted transforms applied once, kept as tensors.

    Items are `(x (1, H_mesh, W_mesh), y (H_taxel, W_taxel), group)`; `y` is in the scaled
    target space. `__getitems__` lets a DataLoader fetch a whole batch by tensor indexing.
    """

    def __init__(
        self,
        paired: PairedData,
        rows: np.ndarray,
        transform_x: Callable[[np.ndarray], np.ndarray] | None = None,
        transform_y: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.rows = np.asarray(rows, dtype=np.int64)
        x = paired.x[self.rows]
        y = paired.y[self.rows]
        if transform_x is not None:
            x = transform_x(x)
        if transform_y is not None:
            y = transform_y(y)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(
            1
        )
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.group = torch.from_numpy(paired.group[self.rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.group[index]

    def __getitems__(self, indices: list[int] | torch.Tensor):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return self.x[idx], self.y[idx], self.group[idx]

    def to(self, device: str | torch.device) -> TaxelDataset:
        self.x, self.y, self.group = (
            t.to(device) for t in (self.x, self.y, self.group)
        )
        return self

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self.x.shape[2], self.x.shape[3]

    @property
    def taxel_shape(self) -> tuple[int, int]:
        return self.y.shape[1], self.y.shape[2]


class GroupBatchSampler(Sampler[list[int]]):
    """Batches of whole groups: every reading of a (force, Path) condition shares a batch.

    `group_of_row[i]` is the group of dataset position i. Each position is yielded exactly
    once per epoch; batches hold `groups_per_batch` groups and a variable number of rows.
    """

    def __init__(
        self,
        group_of_row: np.ndarray,
        groups_per_batch: int,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        group_of_row = np.asarray(group_of_row)
        order = np.argsort(group_of_row, kind="stable")
        _, starts = np.unique(group_of_row[order], return_index=True)
        self.members = np.split(order, starts[1:])
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.members) / self.groups_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        perm = (
            self.rng.permutation(len(self.members))
            if self.shuffle
            else np.arange(len(self.members))
        )
        for start in range(0, len(perm), self.groups_per_batch):
            chunk = perm[start : start + self.groups_per_batch]
            yield np.concatenate([self.members[g] for g in chunk]).tolist()
