"""Tests for taxelnet.ssim_loss and the hybrid alpha * L1 + (1 - alpha) * (1 - SSIM) loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _helpers import write_grasp_pair

from taxelnet import evaluation
from taxelnet.config import ModelConfig, PreprocessingConfig
from taxelnet.dataset import GroupBatchSampler, TaxelDataset, load_paired_files
from taxelnet.model import build_model
from taxelnet.preprocessing import Preprocessor
from taxelnet.ssim_loss import windowed_ssim
from taxelnet.training import train_one_epoch


@pytest.mark.parametrize("window", [(2, 2), (3, 2), (7, 4)])
def test_windowed_ssim_matches_the_evaluation_metric(window):
    rng = np.random.default_rng(0)
    target = rng.normal(300, 400, (5, 7, 4))
    pred = target + rng.normal(0, 150, target.shape)

    got = windowed_ssim(
        torch.from_numpy(pred), torch.from_numpy(target), 3600.0, window
    )

    want = evaluation.windowed_ssim(pred, target, 3600.0, window)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-10)
    same = windowed_ssim(torch.from_numpy(target), torch.from_numpy(target), 3600.0)
    np.testing.assert_allclose(same.numpy(), 1.0)


def test_windowed_ssim_is_differentiable_and_rejects_large_windows():
    target = torch.rand(3, 7, 4) * 1000
    pred = (target + 100 * torch.randn(3, 7, 4)).requires_grad_()
    (1 - windowed_ssim(pred, target, 3600.0)).mean().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="does not fit"):
        windowed_ssim(target, target, 3600.0, (8, 2))


@pytest.mark.parametrize("group_key", ["none", "sensor"])
def test_train_one_epoch_mixes_l1_and_ssim(tmp_path, group_key):
    real, sim = write_grasp_pair(tmp_path, n=16, seed=1, pads=("S1", "S2"))
    paired = load_paired_files(real, sim, ["force", "Name"], None)
    pre = Preprocessor.fit(
        PreprocessingConfig(target_group_key=group_key),
        paired.x,
        paired.y,
        paired.sensor,
    )
    ds = TaxelDataset(
        paired, np.arange(len(paired.group)), pre.transform_x, pre.transform_y
    )
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(base_channels=4, num_down_blocks=1), paired.taxel_shape
    )
    sampler = GroupBatchSampler(ds.group.numpy(), 4, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # weights stay fixed

    pure = train_one_epoch(model, ds, sampler, optimizer, pre=pre)
    mixed = train_one_epoch(model, ds, sampler, optimizer, pre=pre, l1_ssim_alpha=0.25)

    assert "dssim" not in pure and pure["total"] == pytest.approx(pure["l1"])
    assert 0 < mixed["dssim"] <= 2
    assert mixed["total"] == pytest.approx(0.25 * mixed["l1"] + 0.75 * mixed["dssim"])
