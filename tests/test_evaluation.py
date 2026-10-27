"""Tests for taxelnet.evaluation: micro/macro metric correctness and reference points."""

from __future__ import annotations

import numpy as np
import pytest

from taxelnet.evaluation import (
    compute_metrics,
    cosine_similarity,
    global_ssim,
    object_metrics,
    pearson,
    reference_points,
    windowed_ssim,
)


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


def test_decomposed_metrics_perfect_up_to_scale_and_shift():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 300, (6, 7, 4))
    group = np.arange(6)
    m = compute_metrics(2 * target + 40, target, group, ssim_data_range=3600.0)

    assert m["micro_aligned_mae"] == pytest.approx(0, abs=1e-9)
    assert m["micro_peak_hit"] == 1.0 and m["micro_peak_dist"] == 0.0
    assert m["micro_mae"] > 100

    m = compute_metrics(2 * target, target, group, ssim_data_range=3600.0)
    assert m["cosine_median"] == pytest.approx(1.0)
    assert m["amp_logratio_median"] == pytest.approx(np.log(2))
    assert m["amp_logratio_iqr"] == pytest.approx(0, abs=1e-9)
    assert m["gain_aligned_gain"] == pytest.approx(0.5)
    assert m["macro_gain_aligned_mae"] == pytest.approx(0, abs=1e-9)


def test_offset_peak_and_contact():
    target = np.zeros((1, 7, 4))
    target[0, 0, 0] = 100.0  # contact at taxel (0, 0)
    pred = target + 7.0
    pred[0, 3, 2] = 200.0  # predicted peak at (3, 2); contact {(0,0), (3,2)}
    m = compute_metrics(pred, target, np.array([0]), ssim_data_range=3600.0)

    # median residual is 7: only the (3, 2) taxel keeps an error of 200 - 7
    assert m["micro_offset_removed_mae"] == pytest.approx(193 / 28)
    assert m["micro_peak_hit"] == 0.0
    assert m["micro_peak_dist"] == pytest.approx(np.hypot(3, 2))
    assert m["micro_contact_iou"] == pytest.approx(0.5)


def test_object_metrics_groups_and_nocyl():
    target = _maps([10, 20, 30, 40])
    pred = _maps([11, 22, 33, 44])
    names = [
        "Cylinder_TestM_235_x_S1",
        "Cylinder_TestM_235_y_S1",
        "Sphere_TestG_20_a_S1",
        "Cuboid_TestXS_20_b_S1",
    ]
    m = object_metrics(pred, target, np.arange(4), names)

    assert m["group_Cylinder_TestM_235_mae"] == pytest.approx(1.5)
    assert m["group_Sphere_TestG_20_mae"] == pytest.approx(3.0)
    assert m["nocyl_macro_mae"] == pytest.approx(3.5)
    assert m["reference"]["nocyl_macro_mae"] == pytest.approx(35.0)
    assert m["reference"]["group_Cuboid_TestXS_20_mae"] == pytest.approx(40.0)


def test_windowed_ssim_perfect_and_full_window_equals_global():
    rng = np.random.default_rng(0)
    target = rng.normal(500, 300, (5, 7, 4))
    pred = target + rng.normal(0, 100, target.shape)
    assert windowed_ssim(target, target, 3600.0) == pytest.approx(np.ones(5))
    np.testing.assert_allclose(
        windowed_ssim(pred, target, 3600.0, window=(7, 4)),
        global_ssim(pred, target, 3600.0),
    )


def test_windowed_ssim_hand_computed_2x2():
    # 2 x 2 map = a single window; population statistics over its 4 taxels
    pred = np.array([[[0.0, 2.0], [4.0, 6.0]]])
    target = np.array([[[1.0, 1.0], [5.0, 5.0]]])
    L = 10.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mp, mt, vp, vt, cov = 3.0, 3.0, 5.0, 4.0, 4.0
    expected = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp**2 + mt**2 + c1) * (vp + vt + c2)
    )
    assert windowed_ssim(pred, target, L)[0] == pytest.approx(expected)


def test_windowed_ssim_averages_the_6x3_patches():
    rng = np.random.default_rng(1)
    pred, target = rng.random((1, 7, 4)) * 1000, rng.random((1, 7, 4)) * 1000
    patches = [
        global_ssim(
            pred[:, i : i + 2, j : j + 2], target[:, i : i + 2, j : j + 2], 3600.0
        )
        for i in range(6)
        for j in range(3)
    ]
    assert windowed_ssim(pred, target, 3600.0)[0] == pytest.approx(np.mean(patches))
    with pytest.raises(ValueError, match="does not fit"):
        windowed_ssim(pred, target, 3600.0, window=(8, 2))


def test_pearson_and_cosine():
    rng = np.random.default_rng(2)
    target = rng.random((3, 7, 4)) * 1000
    np.testing.assert_allclose(pearson(2 * target + 50, target), 1.0)
    np.testing.assert_allclose(pearson(-target, target), -1.0)
    assert pearson(_maps([5.0]), target[:1])[0] == 0.0  # flat prediction
    np.testing.assert_allclose(cosine_similarity(3 * target, target), 1.0)
    a, b = np.zeros((1, 7, 4)), np.zeros((1, 7, 4))
    a[0, 0, 0], b[0, 0, 1] = 1.0, 1.0
    assert cosine_similarity(a, b)[0] == 0.0  # orthogonal maps
