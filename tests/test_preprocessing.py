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


# --- per-group (sensor) target normalization ---------------------------------------------

SENSOR = np.array(["S1"] * 10 + ["S2"] * 10)
Y_S = Y.copy()
Y_S[10:] = Y_S[10:] * 3.4 + 500  # S2 reads larger, with an offset


@pytest.mark.parametrize("target_scaler", ["global_scale", "per_taxel_zscore"])
def test_group_scaler_fits_each_group_and_round_trips(target_scaler):
    cfg = PreprocessingConfig(target_scaler=target_scaler, target_group_key="sensor")
    pre = Preprocessor.fit(cfg, X, Y_S, SENSOR)
    assert pre.group_labels == ["S1", "S2"]

    scaled = pre.transform_y(Y_S, SENSOR)
    for g in ("S1", "S2"):
        sel = SENSOR == g
        ref = Preprocessor.fit(PreprocessingConfig(target_scaler=target_scaler), X, Y_S[sel])
        np.testing.assert_allclose(scaled[sel], ref.transform_y(Y_S[sel]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pre.inverse_y(scaled, group=SENSOR), Y_S, rtol=1e-4, atol=1e-2)

    t = pre.inverse_y(torch.as_tensor(scaled), group=SENSOR)  # torch path
    np.testing.assert_allclose(t.numpy(), Y_S, rtol=1e-4, atol=1e-2)
    # a prediction is inverted with its own row's parameters
    s1 = pre.inverse_y(scaled[:1], group=np.array(["S2"]))
    assert not np.allclose(s1, Y_S[:1])


def test_group_scaler_json_round_trip(tmp_path):
    cfg = PreprocessingConfig(target_scaler="per_taxel_zscore", target_group_key="sensor")
    pre = Preprocessor.fit(cfg, X, Y_S, SENSOR)
    pre.save(tmp_path / "p.json")
    loaded = Preprocessor.load(tmp_path / "p.json")
    assert loaded.group_labels == ["S1", "S2"]
    np.testing.assert_allclose(loaded.transform_y(Y_S, SENSOR), pre.transform_y(Y_S, SENSOR))
    assert np.asarray(pre.to_dict()["fitted"]["y_scale"]).shape == (2, 7, 4)


def test_group_key_requires_labels_and_known_group():
    cfg = PreprocessingConfig(target_group_key="sensor")
    with pytest.raises(ValueError):
        Preprocessor.fit(cfg, X, Y_S)
    pre = Preprocessor.fit(cfg, X, Y_S, SENSOR)
    with pytest.raises(ValueError):
        pre.transform_y(Y_S)
    with pytest.raises(ValueError):
        pre.transform_y(Y_S[:1], np.array(["S3"]))
    with pytest.raises(ValueError):
        Preprocessor.fit(PreprocessingConfig(target_scaler="asinh", target_group_key="sensor"), X, Y_S, SENSOR)


def test_no_group_key_ignores_labels_and_is_unchanged():
    cfg = PreprocessingConfig(target_scaler="per_taxel_zscore")
    a = Preprocessor.fit(cfg, X, Y_S)
    b = Preprocessor.fit(cfg, X, Y_S, SENSOR)
    assert b.group_labels == []
    np.testing.assert_allclose(a.transform_y(Y_S), b.transform_y(Y_S, SENSOR))
    assert a.to_dict()["fitted"] == b.to_dict()["fitted"] | {}
