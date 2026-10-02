"""Checkpoint save/load helpers.

Not yet implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn, optim


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    raise NotImplementedError


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    raise NotImplementedError
