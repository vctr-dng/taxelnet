"""(object_name, force) join of mesh/taxel GridTables + torch Dataset.

Not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from taxelnet.io_csv import GridTable


@dataclass(frozen=True)
class JoinedSample:
    object_name: str
    force: float
    mesh: np.ndarray
    taxel: np.ndarray


def join_tables(
    mesh: GridTable, taxel: GridTable, strict: bool = True
) -> list[JoinedSample]:
    raise NotImplementedError


class TaxelDataset(Dataset):
    def __init__(self, samples: list[JoinedSample]) -> None:
        raise NotImplementedError

    @classmethod
    def from_csv(
        cls,
        mesh_path: Path,
        taxel_path: Path,
        mesh_grid_shape: tuple[int, int] | None = None,
        taxel_grid_shape: tuple[int, int] | None = None,
        strict: bool = True,
    ) -> TaxelDataset:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def mesh_shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def taxel_shape(self) -> tuple[int, int]:
        raise NotImplementedError


def random_split_dataset(
    dataset: TaxelDataset, val_ratio: float, seed: int
) -> tuple[Subset, Subset]:
    raise NotImplementedError
