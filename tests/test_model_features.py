"""Tests for taxelnet.model_features.FeatureCNN."""

from __future__ import annotations

import pytest
import torch

from taxelnet.config import ModelConfig
from taxelnet.model import CNNTaxelPredictor, build_model
from taxelnet.model_features import DerivedFeatures


@pytest.mark.parametrize(
    "features,convs,norm,base",
    [
        ("z", 1, "none", 16),
        ("z,lap,grad", 1, "none", 16),
        ("z,coord", 1, "none", 16),
        ("z,lap,grad,coord", 2, "group", 32),
    ],
)
def test_feature_cnn_shapes_and_finite(features, convs, norm, base):
    cfg = ModelConfig(
        name="cnn_feat",
        base_channels=base,
        input_features=features,
        convs_per_block=convs,
        norm=norm,
    )
    model = build_model(cfg, (7, 4))
    out = model(torch.randn(3, 1, 77, 49))
    assert out.shape == (3, 7, 4)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_feature_cnn_matches_cnn_v0():
    torch.manual_seed(0)
    v0 = CNNTaxelPredictor((7, 4))
    feat = build_model(ModelConfig(name="cnn_feat"), (7, 4))
    feat.load_state_dict(v0.state_dict())
    x = torch.randn(2, 1, 77, 49)
    assert torch.allclose(v0(x), feat(x))


def test_derived_features_on_flat_and_quadratic_inputs():
    zeros = torch.zeros(1, 1, 9, 9)
    f = DerivedFeatures(("z", "lap", "grad", "coord"))(zeros)
    assert f.shape == (1, 5, 9, 9)
    assert torch.allclose(f[:, :3], torch.zeros_like(f[:, :3]), atol=1e-5)
    # Laplacian of a paraboloid is constant in the interior
    r = torch.arange(9.0)[:, None]
    c = torch.arange(9.0)[None, :]
    lap = DerivedFeatures.laplacian((r**2 + c**2)[None, None])
    inner = lap[0, 0, 1:-1, 1:-1]
    assert torch.allclose(inner, inner[0, 0].expand_as(inner))


def test_cnn_v0_taxel_convs():
    import torch

    from taxelnet.config import ModelConfig
    from taxelnet.model import build_model

    base = build_model(ModelConfig(), (7, 4))
    deep = build_model(ModelConfig(taxel_convs=2), (7, 4))
    x = torch.randn(2, 1, 77, 49)
    assert deep(x).shape == base(x).shape == (2, 7, 4)
    extra = sum(p.numel() for p in deep.parameters()) - sum(
        p.numel() for p in base.parameters()
    )
    assert extra == 2 * (64 * 64 * 9 + 64)
