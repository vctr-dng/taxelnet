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


def test_output_heads_compose_with_physics_models(tmp_path):
    """taxel_affine_head + tanh_bounded act on the final map of every model."""
    x = torch.randn(2, 1, 77, 49).abs() * 0.1
    for name in ("cnn_v0", "cnn_feat", "footprint", "footprint_cnn"):
        cfg = ModelConfig(
            name=name, taxel_affine_head=True, head_activation="tanh_bounded",
            head_lower=-0.5, head_upper=0.5,
        )  # fmt: skip
        model = build_model(cfg, (7, 4))
        keys = set(model.state_dict())
        assert {"taxel_gain", "taxel_offset", "out_act.lo", "out_act.hi"} <= keys
        with torch.no_grad():
            model.taxel_gain.fill_(100.0)
        y = model(x)
        assert y.shape == (2, 7, 4)
        assert (y.abs() <= 0.5).all()  # float32 tanh saturates to exactly 1
        save_model(tmp_path / f"{name}.pt", model, cfg, (7, 4))
        assert torch.allclose(load_model(tmp_path / f"{name}.pt")(x), model.eval()(x))
        default = build_model(ModelConfig(name=name), (7, 4))
        assert not any(k.startswith(("taxel_", "out_act")) for k in default.state_dict())


def test_footprint_cnn_with_coordconv_residual(tmp_path):
    """residual_backbone=cnn_feat: CoordConv CNN branch, zero-init, heads on the sum."""
    from taxelnet.model_features import FeatureCNN

    x = torch.randn(2, 1, 77, 49).abs() * 0.1
    cfg = ModelConfig(
        name="footprint_cnn", residual_backbone="cnn_feat", input_features="z,coord",
        taxel_affine_head=True,
    )  # fmt: skip
    model = build_model(cfg, (7, 4))
    assert isinstance(model.cnn, FeatureCNN)
    assert model.cnn.encoder[0][0].in_channels == 3
    assert not any(k.startswith("cnn.taxel_") for k in model.state_dict())
    y = model(x)
    assert y.shape == (2, 7, 4) and model.residual.abs().max() == 0
    y.sum().backward()
    with torch.no_grad():
        model.cnn.head.weight.normal_()
    save_model(tmp_path / "m.pt", model, cfg, (7, 4))
    assert torch.allclose(load_model(tmp_path / "m.pt")(x), model.eval()(x))
    default = build_model(ModelConfig(name="footprint_cnn"), (7, 4))
    assert type(default.cnn).__name__ == "CNNTaxelPredictor"
