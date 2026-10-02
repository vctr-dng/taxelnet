"""Plain dataclasses for data/model/train configuration + CLI overrides.

Not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    train_mesh_csv: Path
    train_taxel_csv: Path
    test_mesh_csv: Path
    test_taxel_csv: Path
    mesh_grid_shape: tuple[int, int] | None = None
    taxel_grid_shape: tuple[int, int] | None = None
    val_ratio: float = 0.1
    split_seed: int = 0
    strict_join: bool = True


@dataclass
class ModelConfig:
    base_channels: int = 16
    num_down_blocks: int = 3


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs")
    seed: int = 0


def build_arg_parser():
    raise NotImplementedError


def parse_configs(argv=None) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    raise NotImplementedError
