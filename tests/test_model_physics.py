"""Shapes, finiteness and save/load of the physics-structured models."""

from __future__ import annotations

import torch

from taxelnet.checkpoint import load_model, save_model
from taxelnet.config import ModelConfig
from taxelnet.model import build_model


def test_footprint_models_shapes_grads_and_roundtrip(tmp_path):
    x = torch.randn(3, 1, 77, 49).abs() * 0.1
    x[0] = 0.0  # no contact: the power-law bases must stay finite
    x[1, 0, :5] = -0.02  # bulge
    for name in ("footprint", "footprint_cnn"):
        cfg = ModelConfig(name=name)
        model = build_model(cfg, (7, 4))
        y = model(x)
        assert y.shape == (3, 7, 4)
        assert torch.isfinite(y).all()
        y.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        w = (model.footprint if name == "footprint_cnn" else model).footprint_weights()
        assert torch.allclose(w.sum((1, 2)), torch.ones(28), atol=1e-5)
        if name == "footprint_cnn":
            assert model.residual is not None
            assert model.residual.abs().max() == 0  # zero-init residual head
        save_model(tmp_path / f"{name}.pt", model, cfg, (7, 4))
        loaded = load_model(tmp_path / f"{name}.pt")
        assert torch.allclose(loaded(x), model.eval()(x))
        params = model.report(tmp_path)
        assert {"eta", "gamma", "sigma_rows"} <= set(params)
        assert (tmp_path / "footprint_weights.png").exists()
