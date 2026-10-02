"""Reusable evaluate_test_set metrics computation.

Not yet implemented.
"""

from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader


def evaluate_test_set(
    model: nn.Module, test_loader: DataLoader, device: str = "cpu"
) -> dict[str, float]:
    raise NotImplementedError
