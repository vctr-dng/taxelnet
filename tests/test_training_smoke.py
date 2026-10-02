"""End-to-end smoke tests for taxelnet.training.fit and taxelnet.evaluation.

These exercise the reusable library loop directly (not through any scripts/
entrypoint, which doesn't exist yet) on small synthetic in-memory data, so
they stay fast and don't depend on real experimental data.
"""

from __future__ import annotations

import math

import numpy as np
from torch.utils.data import DataLoader

from taxelnet.config import TrainConfig
from taxelnet.dataset import JoinedSample, TaxelDataset, random_split_dataset
from taxelnet.evaluation import evaluate_test_set
from taxelnet.model import CNNTaxelPredictor
from taxelnet.training import fit

MESH_SHAPE = (6, 6)
TAXEL_SHAPE = (2, 2)


def _make_synthetic_samples(n: int, seed: int) -> list[JoinedSample]:
    rng = np.random.default_rng(seed)
    return [
        JoinedSample(
            object_name=f"obj_{i}",
            force=float(i),
            mesh=rng.standard_normal(MESH_SHAPE).astype(np.float32),
            taxel=rng.standard_normal(TAXEL_SHAPE).astype(np.float32),
        )
        for i in range(n)
    ]


def test_fit_runs_two_epochs_and_checkpoints(tmp_path):
    dataset = TaxelDataset(_make_synthetic_samples(12, seed=0))
    train_subset, val_subset = random_split_dataset(dataset, val_ratio=0.25, seed=0)
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4)

    model = CNNTaxelPredictor(taxel_shape=TAXEL_SHAPE)
    train_cfg = TrainConfig(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
        seed=0,
    )

    history = fit(model, train_loader, val_loader, train_cfg)

    assert len(history.train_losses) == 2
    assert len(history.val_losses) == 2
    assert all(math.isfinite(loss) for loss in history.train_losses)
    assert all(math.isfinite(loss) for loss in history.val_losses)
    assert (train_cfg.checkpoint_dir / "last.pt").exists()
    assert (train_cfg.checkpoint_dir / "best.pt").exists()


def test_evaluate_test_set_returns_finite_metrics(tmp_path):
    test_dataset = TaxelDataset(_make_synthetic_samples(5, seed=1))
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = CNNTaxelPredictor(taxel_shape=TAXEL_SHAPE)

    metrics = evaluate_test_set(model, test_loader, device="cpu")

    assert "mse" in metrics
    assert math.isfinite(metrics["mse"])
    assert metrics["mse"] >= 0
