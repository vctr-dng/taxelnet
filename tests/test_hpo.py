"""Tests for taxelnet.hpo: calibrated objectives, search space, end-to-end study smoke."""

from __future__ import annotations

import json
import math

import numpy as np
import optuna
import polars as pl
import pytest
import torch
from _helpers import write_grasp_pair

from taxelnet.config import HPOConfigs, parse_configs
from taxelnet.hpo import (
    OBJECTIVES,
    calibrate_pads,
    group_objectives,
    load_hpo_data,
    objective_metrics,
    reading_objectives,
    run_study,
    suggest_configs,
)
from taxelnet.model import build_model


def test_objective_metrics_calibrates_each_pad():
    rng = np.random.default_rng(0)
    target = rng.random((6, 7, 4)) * 1000 + 10
    pads = np.array(["S1", "S2"] * 3)
    pred = np.where((pads == "S1")[:, None, None], 0.5 * target, 3.0 * target)

    m = objective_metrics(pred, target, pads, 3600.0)

    assert m["gain_S1"] == pytest.approx(2.0)
    assert m["gain_S2"] == pytest.approx(1 / 3)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["ssim"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0) and m["cosine"] == pytest.approx(1.0)
    assert m["raw_mae"] > 100 and m["raw_ssim"] < 1.0

    raw = objective_metrics(pred, target, pads, 3600.0, calibrate=False)
    assert raw["mae"] == pytest.approx(m["raw_mae"])
    assert "gain_S1" not in raw


def test_group_objectives_match_the_whole_set_and_each_group():
    rng = np.random.default_rng(1)
    target = rng.random((8, 7, 4)) * 1000 + 10
    pred = target * rng.uniform(0.5, 1.5, (8, 1, 1))
    pads = np.array(["S1"] * 5 + ["S2"] * 3)
    labels = np.array(["Sphere_TestG_20_S1"] * 5 + ["Cylinder_TestButtons_235_S2"] * 3)
    cal, gains = calibrate_pads(pred, target, pads)
    per_reading = reading_objectives(cal, pred, target, 3600.0)

    by_group = group_objectives(per_reading, labels)

    assert list(by_group) == ["all", *sorted(set(labels))]
    whole = objective_metrics(pred, target, pads, 3600.0)
    for k, v in by_group["all"].items():
        assert v == pytest.approx(whole[k])
    sel = labels == "Sphere_TestG_20_S1"
    assert by_group["Sphere_TestG_20_S1"]["mae"] == pytest.approx(
        np.abs(cal[sel] - target[sel]).mean()
    )
    assert set(gains) == {"gain_S1", "gain_S2"}


@pytest.mark.parametrize("taxel_convs", [0, 2])
def test_suggest_configs_builds_a_standard_cnn(taxel_convs):
    params = {
        "base_channels": 8,
        "num_down_blocks": 4,
        "taxel_convs": taxel_convs,
        "taxel_affine_head": True,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "groups_per_batch": 32,
        "loss_ssi_weight": 0.3,
        "loss_l1_ssim_alpha": 0.4,
        "target_group_key": "sensor",
    }
    if taxel_convs:
        params["taxel_channels"] = 32
    base = HPOConfigs()
    cfgs = suggest_configs(optuna.trial.FixedTrial(params), base)

    assert cfgs.model.name == "cnn_v0" and cfgs.model.input_features == "z"
    assert cfgs.model.taxel_channels == (32 if taxel_convs else 0)
    assert cfgs.train.loss_ssi_weight == 0.3
    assert cfgs.train.loss_l1_ssim_alpha == 0.4
    assert cfgs.train.loss_ssim_window == base.hpo.ssim_window
    assert cfgs.train.loss_ssim_data_range == base.eval.ssim_data_range
    assert cfgs.preprocessing.target_group_key == "sensor"
    assert base.model.base_channels == 16  # base left untouched
    out = build_model(cfgs.model, (7, 4))(torch.zeros(2, 1, 77, 49))
    assert out.shape == (2, 7, 4)


def test_hpo_configs_defaults_and_cli():
    cfgs = parse_configs(["--hpo.n_trials", "3", "--train.epochs", "5"], HPOConfigs)
    assert isinstance(cfgs, HPOConfigs)
    assert cfgs.hpo.n_trials == 3 and cfgs.train.epochs == 5
    assert cfgs.data.real_csv == "Dataset_Grasp_Real_qtac100.csv"
    assert cfgs.data.key_cols == ("force", "Name")
    assert cfgs.hpo.ssim_window == (2, 2)
    assert "hpo.blacklist_csv" in cfgs.flat()


@pytest.fixture
def hpo_cfgs(tmp_path) -> HPOConfigs:
    train_real, train_sim = write_grasp_pair(
        tmp_path / "train", n=24, seed=3, obj="Cuboid_G", pads=("S1", "S2")
    )
    val_real, val_sim = write_grasp_pair(
        tmp_path / "val", n=12, seed=4, obj="Sphere_TestG", pads=("S1", "S2")
    )
    blacklist = tmp_path / "flagged.csv"
    blacklist.write_text("row_in_csv,Name\n2,Cuboid_G_235_0_S1\n")
    cfgs = HPOConfigs()
    d, h = cfgs.data, cfgs.hpo
    d.data_dir, d.real_csv, d.sim_csv = (
        train_real.parent,
        train_real.name,
        train_sim.name,
    )
    d.test_dir, d.test_real_csv, d.test_sim_csv = (
        val_real.parent,
        val_real.name,
        val_sim.name,
    )
    d.cache_dir = None
    h.val_dir, h.val_real_csv, h.val_sim_csv = (
        val_real.parent,
        val_real.name,
        val_sim.name,
    )
    h.blacklist_csv = blacklist
    h.storage, h.n_trials, h.study_name = "memory", 2, "smoke"
    h.output_dir = tmp_path / "hpo"
    cfgs.train.epochs, cfgs.train.device = 2, "cpu"
    cfgs.tracking.uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return cfgs


def test_load_hpo_data_applies_blacklist(hpo_cfgs):
    data = load_hpo_data(hpo_cfgs)
    assert len(data.train.group) == 23 and len(data.val.group) == 12
    assert data.n_blacklisted == {"train": 1, "val": 0, "test": 0}


def test_run_study_smoke(hpo_cfgs):
    pareto = run_study(hpo_cfgs)

    out = hpo_cfgs.hpo.output_dir / "smoke"
    trials = pl.read_csv(out / "trials.csv")
    assert len(trials) == 2 and trials["pareto"].any()
    assert len(pareto) == trials["pareto"].sum()
    for k in OBJECTIVES:
        assert all(math.isfinite(v) for v in pareto[f"val_{k}"])
        # test = the val files here, so the saved models reproduce the val objectives
        np.testing.assert_allclose(pareto[f"test_{k}"], pareto[f"val_{k}"], rtol=1e-5)
    for name in ("pareto.csv", "pareto.png", "study.json"):
        assert (out / name).exists()
    seed_dir = out / "trial_0000" / "seed0"
    for name in ("model.pt", "preprocessing.json", "val_metrics.json", "last.pt"):
        assert (seed_dir / name).exists()
    study = json.loads((out / "study.json").read_text())
    assert study["readings"] == {"train": 23, "val": 12, "test": 12}
