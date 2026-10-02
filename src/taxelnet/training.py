"""Reusable training loop: train_one_epoch / evaluate_loss / fit.

Not yet implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from taxelnet.config import TrainConfig


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> float:
    raise NotImplementedError


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> float:
    raise NotImplementedError


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
) -> TrainingHistory:
    raise NotImplementedError
