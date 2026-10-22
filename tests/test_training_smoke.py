"""End-to-end smoke test: two epochs of group-balanced training with local MLflow tracking.

Runs the library orchestration (`run_training`, what scripts/train.py calls) on a small
synthetic matched pair, so it stays fast and doesn't depend on real experimental data.
"""

from __future__ import annotations

import json
import math

import mlflow
import numpy as np
import polars as pl
import pytest
import torch

from taxelnet.checkpoint import load_checkpoint, load_model
from taxelnet.config import Configs, parse_configs
from taxelnet.dataset import TaxelDataset, load_test_paired
from taxelnet.evaluation import predict
from taxelnet.model import build_model
from taxelnet.preprocessing import Preprocessor
from taxelnet.training import group_balanced_l1, run_training

BUNDLE = [
    "model.pt",
    "best.pt",
    "last.pt",
    "preprocessing.json",
    "node_ids.csv",
    "schema_report.json",
    "split_manifest.json",
    "metrics.json",
    "predictions_test.parquet",
    "requirements.lock.txt",
    "config.json",
    "checkpoints/epoch_0002.pt",
    "checkpoints/epoch_0003.pt",
    "plots/epoch_0002/samples.png",
    "plots/epoch_0002/scatter.png",
    "plots/epoch_0002/per_taxel_mae.png",
    "plots/epoch_0003/test_metrics.json",
    "plots/best/samples.png",
    "plots/best/ssim_by_group.png",
    "plots/best_val/ssim_by_group.png",
]


def test_group_balanced_l1_averages_groups_equally():
    pred = torch.tensor([[1.0], [1.0], [1.0], [5.0]])
    target = torch.zeros(4, 1)
    group = torch.tensor([7, 7, 7, 2])

    assert group_balanced_l1(pred, target, group).item() == pytest.approx(3.0)
    assert group_balanced_l1(pred, target, group, alpha=1.0).item() == pytest.approx(
        2.0
    )


def test_parse_configs_overrides():
    cfgs = parse_configs(
        [
            "--train.epochs",
            "2",
            "--data.cache_dir",
            "none",
            "--preprocessing.clip_input=-1,3",
        ]
    )
    assert cfgs.train.epochs == 2
    assert cfgs.data.cache_dir is None
    assert cfgs.preprocessing.clip_input == (-1.0, 3.0)
    assert Configs().train.epochs != 2


def test_run_training_two_epochs(tmp_path, synthetic_pair, grasp_pair):
    real_csv, sim_csv = synthetic_pair
    cfgs = Configs()
    cfgs.data.data_dir = tmp_path
    cfgs.data.real_csv, cfgs.data.sim_csv = real_csv.name, sim_csv.name
    cfgs.data.cache_dir = tmp_path / "cache"
    cfgs.data.test_dir = grasp_pair[0].parent
    cfgs.data.test_real_csv, cfgs.data.test_sim_csv = (f.name for f in grasp_pair)
    cfgs.train.epochs = 3
    cfgs.train.test_every = 2  # epoch 2, plus the last epoch
    cfgs.train.groups_per_batch = 8
    cfgs.train.device = "cpu"
    cfgs.train.output_dir = tmp_path / "outputs"
    cfgs.tracking.uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfgs.tracking.artifact_location = (tmp_path / "mlartifacts").as_uri()

    result = run_training(cfgs)

    for name in BUNDLE:
        assert (result.run_dir / name).exists(), name
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert math.isfinite(metrics["test"]["micro_mae"])
    assert math.isfinite(metrics["test"]["macro_mae"])
    assert metrics["integrity"]["scalers_fitted_on"] == "train"
    assert metrics["test"]["n_readings"] == 20

    preds = pl.read_parquet(result.run_dir / "predictions_test.parquet")
    assert preds.height == metrics["test"]["n_readings"]
    assert set(preds["partition"]) == {"test"}

    client = mlflow.MlflowClient(cfgs.tracking.uri)
    run = client.get_run(result.run_id)
    assert {"train_loss", "val_macro_mae", "final_test_macro_mae"} <= set(
        run.data.metrics
    )
    assert len(client.get_metric_history(result.run_id, "val_micro_mae")) == 3
    test_history = client.get_metric_history(result.run_id, "test_micro_mae")
    assert sorted(m.step for m in test_history) == [2, 3]
    assert not (result.run_dir / "checkpoints" / "epoch_0001.pt").exists()
    assert run.data.params["train.epochs"] == "3"
    assert "sha256." + real_csv.name in run.data.tags
    assert "sha256." + grasp_pair[0].name in run.data.tags
    assert run.data.params["split.test_readings"] == "20"
    artifacts = {a.path for a in client.list_artifacts(result.run_id)}
    assert {"best.pt", "metrics.json", "split_manifest.json", "plots"} <= artifacts
    assert {a.path for a in client.list_artifacts(result.run_id, "plots")} == {
        "plots/epoch_0002",
        "plots/epoch_0003",
        "plots/best",
        "plots/best_val",
    }
    assert {a.path for a in client.list_artifacts(result.run_id, "checkpoints")} == {
        "checkpoints/epoch_0002.pt",
        "checkpoints/epoch_0003.pt",
    }

    # bundle round trip: best.pt and model.pt reproduce the evaluated predictions
    pre = Preprocessor.load(result.run_dir / "preprocessing.json")
    test = load_test_paired(cfgs.data)
    test_rows = np.sort(preds["source_row"].to_numpy())
    np.testing.assert_array_equal(test_rows, np.arange(len(test.group)))
    ds = TaxelDataset(test, test_rows, pre.transform_x, pre.transform_y)
    model = build_model(cfgs.model, test.taxel_shape)
    ckpt = load_checkpoint(result.run_dir / "best.pt", model)
    assert ckpt["mlflow_run_id"] == result.run_id
    assert ckpt["epoch"] == metrics["best_epoch"]
    expected = np.array(preds.sort("source_row")["prediction"].to_list())
    expected = expected.reshape(-1, 7, 4)
    np.testing.assert_allclose(predict(model, ds, pre), expected, rtol=1e-5, atol=1e-3)
    loaded = load_model(result.run_dir / "model.pt")
    np.testing.assert_allclose(predict(loaded, ds, pre), expected, rtol=1e-5, atol=1e-3)
