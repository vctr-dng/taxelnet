"""Scale-and-shift-invariant pattern losses (taxelnet.ssi)."""

from __future__ import annotations

import torch

from taxelnet.ssi import ssi_losses, ssi_normalize


def test_ssi_is_invariant_to_affine_change_of_target():
    torch.manual_seed(0)
    target = torch.randn(16, 7, 4) * 3
    pred = 2.5 * target + 1.7  # same pattern, other gain and offset
    l_ssi, l_grad = ssi_losses(pred, target, eps=1e-6)
    assert l_ssi.max().item() < 1e-4
    assert l_grad.max().item() < 1e-4
    # a different pattern is penalized
    l_ssi, l_grad = ssi_losses(torch.randn(16, 7, 4), target, eps=1e-6)
    assert (l_ssi > 0.1).all() and (l_grad > 0.1).all()


def test_normalized_map_has_zero_median_and_unit_mad():
    d = torch.randn(8, 7, 4) * 10 + 5
    n = ssi_normalize(d, eps=0.0).flatten(1)
    assert torch.allclose(n.median(1).values, torch.zeros(8), atol=1e-5)
    assert torch.allclose(n.abs().mean(1), torch.ones(8), atol=1e-5)


def test_flat_maps_give_bounded_gradients():
    eps = 0.05
    target = torch.zeros(4, 7, 4)
    pred = (1e-6 * torch.randn(4, 7, 4)).requires_grad_()
    l_ssi, l_grad = ssi_losses(pred, target, eps)
    assert torch.isfinite(l_ssi).all() and l_ssi.max() < 1e-3
    (l_ssi.sum() + l_grad.sum()).backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max() <= 4 * 2 / eps  # bounded by ~1/eps per term
