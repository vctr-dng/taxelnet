"""Tests for taxelnet.amp: asinh round trip/clamp and the log-amplitude loss."""

from __future__ import annotations

import math

import numpy as np
import torch

from taxelnet.amp import logamp_per_press
from taxelnet.config import PreprocessingConfig
from taxelnet.preprocessing import Preprocessor

rng = np.random.default_rng(0)
X = (rng.random((20, 77, 49)) * 2.4 - 0.1).astype(np.float32)
Y = rng.normal(200, 300, (20, 7, 4)).astype(np.float32)


def _asinh(scale=30.0, margin=0.25):
    cfg = PreprocessingConfig(
        target_scaler="asinh", asinh_scale=scale, asinh_clamp_margin=margin
    )
    return Preprocessor.fit(cfg, X, Y)


def test_asinh_round_trip_numpy_torch_and_json(tmp_path):
    pre = _asinh()
    np.testing.assert_allclose(pre.inverse_y(pre.transform_y(Y)), Y, rtol=1e-4, atol=1e-2)
    t = torch.from_numpy(Y)
    back = pre.inverse_y(pre.transform_y(t))
    assert back.dtype == torch.float32
    torch.testing.assert_close(back, t, rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(pre.transform_y(Y), np.arcsinh(Y / 30.0), rtol=1e-6)

    pre.save(tmp_path / "pre.json")
    loaded = Preprocessor.load(tmp_path / "pre.json")
    assert (loaded.y_clamp_lo, loaded.y_clamp_hi) == (pre.y_clamp_lo, pre.y_clamp_hi)
    np.testing.assert_allclose(loaded.inverse_y(pre.transform_y(Y)), Y, rtol=1e-4, atol=1e-2)


def test_asinh_inverse_clamps_extrapolation():
    pre = _asinh(scale=30.0, margin=0.25)
    lo, hi = np.arcsinh(Y.min() / 30.0) - 0.25, np.arcsinh(Y.max() / 30.0) + 0.25
    assert math.isclose(pre.y_clamp_lo, lo, rel_tol=1e-6)
    assert math.isclose(pre.y_clamp_hi, hi, rel_tol=1e-6)
    huge = torch.tensor([[50.0, -50.0]])
    out = pre.inverse_y(huge)
    torch.testing.assert_close(
        out, torch.tensor([[30 * math.sinh(hi), 30 * math.sinh(lo)]]), rtol=1e-5, atol=1e-3
    )
    # unclamped inverse (used in the training loss) keeps gradients alive
    assert pre.inverse_y(torch.tensor([10.0]), clamp=False).item() > out.max().item()


def test_global_scale_unchanged_by_clamp_argument():
    pre = Preprocessor.fit(PreprocessingConfig(), X, Y)
    t = torch.from_numpy(Y)
    torch.testing.assert_close(pre.inverse_y(t), pre.inverse_y(t, clamp=False))
    assert pre.y_clamp_lo is None and pre.y_clamp_hi is None


def test_logamp_is_scale_error_in_log_space():
    y = torch.rand(5, 7, 4) * 1000
    # eps -> 0: exactly |log gain|, independent of the pattern
    out = logamp_per_press(1.5 * y, y, eps=1e-9)
    torch.testing.assert_close(out, torch.full((5,), math.log(1.5)), rtol=1e-5, atol=1e-5)
    assert torch.allclose(logamp_per_press(y, y, eps=50.0), torch.zeros(5))
    # permuting taxels keeps the norm: amplitude only, not pattern
    perm = y.flatten(1)[:, torch.randperm(28)].view_as(y)
    assert torch.allclose(logamp_per_press(perm, y, eps=50.0), torch.zeros(5), atol=1e-6)
