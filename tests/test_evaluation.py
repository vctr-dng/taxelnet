"""Tests for taxelnet.evaluation: micro/macro metric correctness and reference points."""

from __future__ import annotations

import numpy as np
import pytest

from taxelnet.evaluation import compute_metrics, global_ssim, reference_points


def _maps(values):
    return np.stack([np.full((7, 4), v, dtype=np.float32) for v in values])


def test_micro_and_macro_mae_differ_on_repeated_group():
    # group 0 has 3 readings with error 10, group 1 one reading with error 50
    target = _maps([0, 0, 0, 0])
    pred = _maps([10, -10, 10, 50])
    group = np.array([0, 0, 0, 1])

    m = compute_metrics(pred, target, group, ssim_data_range=3600.0)

    assert m["micro_mae"] == pytest.approx((10 * 3 + 50) / 4)
    assert m["macro_mae"] == pytest.approx((10 + 50) / 2)
    assert m["micro_mse"] == pytest.approx((100 * 3 + 2500) / 4)
    assert m["macro_rmse"] == pytest.approx(np.sqrt((100 + 2500) / 2))
    assert m["micro_crps"] == m["micro_mae"]
    assert m["micro_energy_score"] == pytest.approx((3 * 10 + 50) * np.sqrt(28) / 4)
    assert len(m["per_taxel_micro_mae"]) == 28
    assert m["per_taxel_macro_mae"][0] == pytest.approx(30.0)
    assert (m["n_readings"], m["n_groups"]) == (4, 2)


def test_ssim_is_one_for_perfect_prediction():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 500, (5, 7, 4))
    np.testing.assert_allclose(global_ssim(target, target, 3600.0), 1.0)
    assert (global_ssim(target + 300, target, 3600.0) < 1).all()


def test_reference_points():
    target = np.stack(
        [
            np.full((7, 4), 10.0),
            np.full((7, 4), 20.0),
            np.full((7, 4), 60.0),  # group 0 median = 20 -> errors 10, 0, 40
            np.full((7, 4), -100.0),  # singleton: excluded from the noise floor
        ]
    )
    group = np.array([3, 3, 3, 8])

    ref = reference_points(target, group)

    assert ref["noise_floor"]["micro_mae"] == pytest.approx(50 / 3)
    assert ref["noise_floor"]["macro_mae"] == pytest.approx(50 / 3)
    assert ref["noise_floor"]["n_repeated_groups"] == 1
    assert ref["trivial_zero"]["micro_mae"] == pytest.approx(190 / 4)
    assert ref["trivial_zero"]["macro_mae"] == pytest.approx((30 + 100) / 2)
