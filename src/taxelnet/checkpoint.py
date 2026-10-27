"""Checkpoint (best.pt / last.pt) and inference model (model.pt) save/load helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from taxelnet.config import ModelConfig
from taxelnet.model import build_model

ARTIFACT_CONTRACT_VERSION = "0.1"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Model + optimizer state, epoch and contract version; `extra` holds the model
    config, best validation metric and MLflow run id."""
    torch.save(
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("artifact_contract_version") != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            f"{path}: artifact contract {ckpt.get('artifact_contract_version')} != "
            f"{ARTIFACT_CONTRACT_VERSION}"
        )
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def save_model(
    path: Path, model: nn.Module, model_cfg: ModelConfig, taxel_shape: tuple[int, int]
) -> None:
    """Inference bundle entry: weights + what is needed to rebuild the module."""
    torch.save(
        {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "model_config": asdict(model_cfg),
            "taxel_shape": list(taxel_shape),
            "model_state": model.state_dict(),
        },
        path,
    )


def load_model(path: Path, map_location: str = "cpu") -> nn.Module:
    data = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ModelConfig(**data["model_config"]), tuple(data["taxel_shape"]))
    model.load_state_dict(data["model_state"])
    return model.eval()
