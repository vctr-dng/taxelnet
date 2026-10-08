"""Tests for taxelnet.preprocessing: train-only fitting, invertibility, JSON round trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from taxelnet.config import PreprocessingConfig
from taxelnet.preprocessing import INPUT_SCALERS, TARGET_SCALERS, Preprocessor

rng = np.random.default_rng(0)
X = (rng.random((20, 77, 49)) * 2.4 - 0.1).astype(np.float32)
Y = (rng.normal(200, 300, (20, 7, 4))).astype(np.float32)
TRAIN = np.arange(14)


@pytest.mark.parametrize("input_scaler", INPUT_SCALERS)
@pytest.mark.parametrize("target_scaler", TARGET_SCALERS)
def test_statistics_come_from_train_rows_only(input_scaler, target_scaler):
    cfg = PreprocessingConfig(input_scaler=input_scaler, target_scaler=target_scaler)
    pre = Preprocessor.fit(cfg, X[TRAIN], Y[TRAIN])

    x2, y2 = X.copy(), Y.copy()
    x2[14:] *= 100  # perturb validation/test rows
    y2[14:] -= 5000
    pre2 = Preprocessor.fit(cfg, x2[TRAIN], y2[TRAIN])

    assert pre.to_dict()["fitted"] == pre2.to_dict()["fitted"]
    assert pre.fitted_on == "train" and pre.n_fit_rows == len(TRAIN)
    # the same fitted transform is applied unchanged to held-out rows
    np.testing.assert_allclose(pre.transform_x(x2[14:]), pre2.transform_x(x2[14:]))


@pytest.mark.parametrize("target_scaler", TARGET_SCALERS)
def test_inverse_y_round_trip(target_scaler):
    pre = Preprocessor.fit(PreprocessingConfig(target_scaler=target_scaler), X, Y)

    np.testing.assert_allclose(
        pre.inverse_y(pre.transform_y(Y)), Y, rtol=1e-5, atol=1e-3
    )
    t = torch.from_numpy(Y)
    back = pre.inverse_y(pre.transform_y(t))
    assert isinstance(back, torch.Tensor) and back.dtype == torch.float32
    torch.testing.assert_close(back, t, rtol=1e-5, atol=1e-3)


def test_global_scale_keeps_l1_proportional_to_raw():
    pre = Preprocessor.fit(PreprocessingConfig(target_scaler="global_scale"), X, Y)
    pred = Y + rng.normal(0, 50, Y.shape).astype(np.float32)
    raw = np.abs(pred - Y).mean()
    scaled = np.abs(pre.transform_y(pred) - pre.transform_y(Y)).mean()
    assert scaled * float(pre.y_scale) == pytest.approx(raw, rel=1e-4)


def test_max_abs_keeps_negative_bulge():
    pre = Preprocessor.fit(PreprocessingConfig(input_scaler="max_abs"), X, Y)
    assert pre.transform_x(X).min() < 0
    assert np.abs(pre.transform_x(X)).max() == pytest.approx(1.0)


def test_clip_input():
    cfg = PreprocessingConfig(input_scaler="identity", clip_input=(0.0, 1.0))
    pre = Preprocessor.fit(cfg, X, Y)
    out = pre.transform_x(X)
    assert out.min() == 0.0 and out.max() == 1.0


def test_json_round_trip(tmp_path):
    cfg = PreprocessingConfig("robust_zscore", "per_taxel_zscore", (-0.2, 3.0))
    pre = Preprocessor.fit(cfg, X, Y)
    pre.save(tmp_path / "preprocessing.json")
    loaded = Preprocessor.load(tmp_path / "preprocessing.json")

    assert loaded.config == cfg
    np.testing.assert_allclose(loaded.transform_x(X), pre.transform_x(X))
    np.testing.assert_allclose(loaded.transform_y(Y), pre.transform_y(Y))
    contract = pre.to_dict()["contract"]
    assert contract["input"]["units"] == "mm"
    assert contract["input"]["grid_shape"] == [77, 49]


def test_unknown_scaler_rejected():
    with pytest.raises(ValueError, match="input_scaler"):
        Preprocessor.fit(PreprocessingConfig(input_scaler="minmax"), X, Y)
