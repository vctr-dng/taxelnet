from __future__ import annotations

import numpy as np

from taxelnet.plots import plot_group_label, save_inference_plots, select_group_samples


def test_plot_group_label():
    assert (
        plot_group_label("Cylinder_TestButtons_235_15_A_V+2_H-1_Deg60_S2", 235.0)
        == "Cylinder_TestButtons_235_S2"
    )
    assert (
        plot_group_label("Cuboid_G/20_02_A_V+1_H+1_Deg15_S1", 20.0) == "Cuboid_G_20_S1"
    )
    assert plot_group_label("TriangleM/Lateral2", 40.0) == "TriangleM_40"


def test_select_group_samples_stays_in_group():
    target = np.arange(10, dtype=float).reshape(10, 1, 1)
    labels = np.array(["a"] * 6 + ["b"] * 4)
    picks = select_group_samples(target, labels, 3)
    assert set(picks) == {"a", "b"}
    assert all((labels[p] == g).all() and len(p) == 3 for g, p in picks.items())
    assert picks["a"].tolist() == [0, 2, 5]  # weak to strong


def test_save_inference_plots_per_group(tmp_path):
    rng = np.random.default_rng(0)
    n = 8
    x, y = rng.normal(size=(n, 5, 4)), rng.normal(size=(n, 7, 4))
    pred = y + rng.normal(scale=0.1, size=y.shape)
    metrics = {
        "micro_mae": 0.1, "macro_mae": 0.1, "micro_rmse": 0.1,
        "per_taxel_micro_mae": [0.1] * 28,
    }  # fmt: skip
    labels = np.array(["Sphere_TestG_20_S1"] * 5 + ["Cylinder_TestButtons_235_S2"] * 3)
    save_inference_plots(
        tmp_path, x, y, pred, metrics, np.array([0, 7]), ["r0", "r7"], "t",
        group_labels=labels, ssim=rng.uniform(size=n),
        titles=[f"row {i}" for i in range(n)], per_group=2,
    )  # fmt: skip
    assert (tmp_path / "ssim_by_group.png").exists()
    assert (tmp_path / "samples" / "Sphere_TestG_20_S1.png").exists()
    assert (tmp_path / "samples" / "Cylinder_TestButtons_235_S2.png").exists()
    assert (tmp_path / "samples.png").exists()
