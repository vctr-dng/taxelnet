import numpy as np

from taxelnet.calibration import (
    fit_calibration,
    run_protocol,
    shots_to_oracle,
    weighted_median,
)


def _data(n=40, t=28, gain=0.4, offset=-25.0, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.gamma(1.0, 300.0, size=(n, t))
    target = gain * pred + offset + rng.laplace(0, 5, size=(n, t))
    return pred, target


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0


def test_global_fits_recover_gain_and_offset():
    pred, target = _data()
    g = fit_calibration("global_gain", pred[:, :], target - (-25.0))
    assert abs(float(g.gain) - 0.4) < 0.02
    a = fit_calibration("global_affine", pred, target)
    assert abs(float(a.gain) - 0.4) < 0.01 and abs(float(a.offset) + 25) < 2


def test_global_affine_is_robust_to_outliers():
    pred, target = _data()
    target[0] += 5000.0
    a = fit_calibration("global_affine", pred, target)
    assert abs(float(a.gain) - 0.4) < 0.01


def test_taxel_affine_shrinks_and_recovers():
    rng = np.random.default_rng(1)
    gains = rng.uniform(0.3, 0.6, 28)
    pred, _ = _data(n=200)
    target = gains * pred + rng.laplace(0, 5, size=pred.shape)
    c = fit_calibration("taxel_affine", pred, target, tau=1.0)
    assert np.abs(c.gain - gains).max() < 0.05
    # one reading + a strong prior: close to the global gain
    c1 = fit_calibration("taxel_affine", pred[:1], target[:1], tau=1e6)
    glob = fit_calibration("global_affine", pred[:1], target[:1])
    assert np.allclose(c1.gain, float(glob.gain), atol=1e-3)


def test_protocol_shapes():
    pred, target = _data(n=30)
    group = np.arange(30)
    res = run_protocol(pred, target, group, ks=(1, 3), n_draws=3)
    assert res["global_gain"]["all"]["gain"] > 0
    assert res["none"]["1"]["n_draws"] == 3
    assert res["global_affine"]["3"]["macro_mae_mean"] < res["none"]["3"]["macro_mae_mean"]
    assert shots_to_oracle(res, "global_affine") in (1, 3)


def test_taxel_affine_head_is_identity_at_init():
    import torch

    from taxelnet.model import CNNTaxelPredictor

    torch.manual_seed(0)
    base = CNNTaxelPredictor((7, 4))
    torch.manual_seed(0)
    head = CNNTaxelPredictor((7, 4), taxel_affine_head=True)
    x = torch.randn(2, 1, 77, 49)
    assert torch.allclose(base(x), head(x))
    assert sum(p.numel() for p in head.parameters()) - sum(
        p.numel() for p in base.parameters()
    ) == 56
