"""Tests for taxelnet.model.CNNTaxelPredictor."""

from __future__ import annotations

import torch

from taxelnet.config import ModelConfig
from taxelnet.model import CNNTaxelPredictor, build_model


def test_cnn_taxel_predictor_output_shape_matches_taxel_grid():
    taxel_shape = (2, 3)
    model = CNNTaxelPredictor(taxel_shape=taxel_shape)

    x = torch.randn(3, 1, 16, 20)  # mesh grid much finer than taxel grid
    out = model(x)

    assert out.shape == (3, *taxel_shape)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_cnn_taxel_predictor_output_size_independent_of_resolution_ratio():
    """AdaptiveAvgPool2d bridge should force exact output size even when the
    mesh/taxel resolution ratio doesn't divide evenly."""
    taxel_shape = (2, 3)
    model = CNNTaxelPredictor(taxel_shape=taxel_shape)

    x = torch.randn(1, 1, 13, 17)  # awkward, non-multiple input resolution
    out = model(x)

    assert out.shape == (1, *taxel_shape)


def test_cnn_taxel_predictor_single_sample_batch():
    taxel_shape = (4, 4)
    model = CNNTaxelPredictor(taxel_shape=taxel_shape)

    x = torch.randn(1, 1, 32, 32)
    out = model(x)

    assert out.shape == (1, *taxel_shape)


def test_cnn_taxel_predictor_on_indenter_grids():
    model = build_model(ModelConfig(), (7, 4))
    out = model(torch.randn(2, 1, 77, 49))

    assert out.shape == (2, 7, 4)
    assert torch.isfinite(out).all()
