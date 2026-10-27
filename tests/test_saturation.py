"""Tests for taxelnet.saturation: bounded head and asinh input scaler."""

from __future__ import annotations

import numpy as np
import torch

from taxelnet.checkpoint import load_model, save_model
from taxelnet.config import ModelConfig, PreprocessingConfig
from taxelnet.model import build_model
from taxelnet.preprocessing import Preprocessor
from taxelnet.saturation import BoundedTanh, head_bounds


def test_bounded_tanh_stays_within_bounds_and_is_identity_near_zero():
    act = BoundedTanh(-0.5, 2.0)
    z = torch.linspace(-1e4, 1e4, 20001)
    y = act(z)
    assert y.min() >= -0.5 and y.max() <= 2.0
    small = torch.tensor([-1e-3, 1e-3])
    torch.testing.assert_close(act(small), small, rtol=1e-5, atol=1e-8)
    assert torch.all(torch.diff(y) >= 0)  # monotone


def test_head_bounds_from_train_range():
    lo, hi = head_bounds(np.array([-1.0, 0.5, 3.0]), 1.1)
    assert np.isclose(lo, -1.1) and np.isclose(hi, 3.3)
    lo, hi = head_bounds(np.array([0.0, 2.0]), 1.1)  # no negatives: still lo < 0
    assert lo < 0 < hi


def test_bounded_model_output_and_reload(tmp_path):
    cfg = ModelConfig(head_activation="tanh_bounded", head_lower=-0.3, head_upper=0.7)
    model = build_model(cfg, (7, 4))
    with torch.no_grad():
        model.head.bias.fill_(1e3)
        out = model(torch.randn(2, 1, 77, 49))
    assert (out <= 0.7 + 1e-6).all() and (out >= -0.3 - 1e-6).all()
    save_model(tmp_path / "model.pt", model, cfg, (7, 4))
    loaded = load_model(tmp_path / "model.pt")
    assert np.isclose(loaded.out_act.lo.item(), -0.3)
    assert np.isclose(loaded.out_act.hi.item(), 0.7)
    x = torch.randn(2, 1, 77, 49)
    torch.testing.assert_close(loaded(x), model.eval()(x))


def test_linear_default_has_no_extra_state():
    model = build_model(ModelConfig(), (7, 4))
    assert not any(k.startswith("out_act") for k in model.state_dict())


def test_asinh_input_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.random((10, 77, 49)) * 1.2 - 0.05).astype(np.float32)
    y = rng.normal(0, 300, (10, 7, 4)).astype(np.float32)
    cfg = PreprocessingConfig(input_scaler="asinh", input_asinh_scale=0.2)
    pre = Preprocessor.fit(cfg, x, y)
    xs = pre.transform_x(x)
    assert np.isclose(np.abs(xs).max(), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        xs, np.arcsinh(x / 0.2) / np.arcsinh(np.abs(x).max() / 0.2), rtol=1e-5
    )
    # compressive: a 4x deeper contact gives much less than 4x the input
    a = pre.transform_x(np.array([0.25], dtype=np.float32))
    b = pre.transform_x(np.array([1.0], dtype=np.float32))
    assert b / a < 2.5
    pre.save(tmp_path / "pre.json")
    pre2 = Preprocessor.load(tmp_path / "pre.json")
    assert pre2.config.input_asinh_scale == 0.2
    np.testing.assert_allclose(pre2.transform_x(x), xs)
    np.testing.assert_allclose(
        pre2.transform_x(torch.from_numpy(x)).numpy(), xs, rtol=1e-5, atol=1e-6
    )
