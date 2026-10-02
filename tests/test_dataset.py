"""Tests for taxelnet.dataset: (object_name, force) join + torch Dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _helpers import (
    MESH_GRID_SHAPE,
    SAMPLE_KEYS,
    TAXEL_GRID_SHAPE,
    expected_grid_array,
    mesh_value,
    taxel_value,
)

from taxelnet.dataset import (
    JoinedSample,
    TaxelDataset,
    join_tables,
    random_split_dataset,
)
from taxelnet.io_csv import read_grid_csv


def test_join_tables_matches_on_object_name_and_force(tiny_mesh_csv, tiny_taxel_csv):
    mesh = read_grid_csv(tiny_mesh_csv)
    taxel = read_grid_csv(tiny_taxel_csv)

    samples = join_tables(mesh, taxel, strict=True)

    assert len(samples) == len(SAMPLE_KEYS)
    expected_mesh = expected_grid_array(SAMPLE_KEYS, MESH_GRID_SHAPE, mesh_value)
    expected_taxel = expected_grid_array(SAMPLE_KEYS, TAXEL_GRID_SHAPE, taxel_value)
    for i, (object_name, force) in enumerate(SAMPLE_KEYS):
        sample = samples[i]
        assert sample.object_name == object_name
        assert sample.force == pytest.approx(float(force))
        np.testing.assert_allclose(sample.mesh, expected_mesh[i])
        np.testing.assert_allclose(sample.taxel, expected_taxel[i])


def test_join_tables_strict_raises_on_mismatched_keys(
    tiny_mesh_csv, mismatched_taxel_csv
):
    mesh = read_grid_csv(tiny_mesh_csv)
    taxel = read_grid_csv(mismatched_taxel_csv)

    with pytest.raises(ValueError):
        join_tables(mesh, taxel, strict=True)


def test_join_tables_non_strict_drops_unmatched_keys(
    tiny_mesh_csv, mismatched_taxel_csv
):
    mesh = read_grid_csv(tiny_mesh_csv)
    taxel = read_grid_csv(mismatched_taxel_csv)

    samples = join_tables(mesh, taxel, strict=False)

    # mesh keys: obj_a/1.0, obj_a/2.0, obj_b/1.0, obj_b/2.0
    # taxel keys: obj_a/1.0, obj_a/2.0, obj_b/1.0, obj_c/1.0
    # intersection: obj_a/1.0, obj_a/2.0, obj_b/1.0
    got_keys = {(s.object_name, s.force) for s in samples}
    assert got_keys == {("obj_a", 1.0), ("obj_a", 2.0), ("obj_b", 1.0)}


def test_taxel_dataset_from_csv_shapes_and_tensors(tiny_mesh_csv, tiny_taxel_csv):
    dataset = TaxelDataset.from_csv(tiny_mesh_csv, tiny_taxel_csv)

    assert len(dataset) == len(SAMPLE_KEYS)
    assert dataset.mesh_shape == MESH_GRID_SHAPE
    assert dataset.taxel_shape == TAXEL_GRID_SHAPE

    mesh_tensor, taxel_tensor = dataset[0]

    assert isinstance(mesh_tensor, torch.Tensor)
    assert isinstance(taxel_tensor, torch.Tensor)
    assert mesh_tensor.shape == (1, *MESH_GRID_SHAPE)
    assert taxel_tensor.shape == tuple(TAXEL_GRID_SHAPE)
    assert mesh_tensor.dtype == torch.float32
    assert taxel_tensor.dtype == torch.float32

    expected_mesh = expected_grid_array(SAMPLE_KEYS, MESH_GRID_SHAPE, mesh_value)[0]
    expected_taxel = expected_grid_array(SAMPLE_KEYS, TAXEL_GRID_SHAPE, taxel_value)[0]
    np.testing.assert_allclose(mesh_tensor.numpy()[0], expected_mesh)
    np.testing.assert_allclose(taxel_tensor.numpy(), expected_taxel)


def test_taxel_dataset_from_csv_strict_mismatch_raises(
    tiny_mesh_csv, mismatched_taxel_csv
):
    with pytest.raises(ValueError):
        TaxelDataset.from_csv(tiny_mesh_csv, mismatched_taxel_csv, strict=True)


def _make_synthetic_samples(
    n: int, mesh_shape: tuple[int, int], taxel_shape: tuple[int, int], seed: int = 0
) -> list[JoinedSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        samples.append(
            JoinedSample(
                object_name=f"obj_{i}",
                force=float(i),
                mesh=rng.standard_normal(mesh_shape).astype(np.float32),
                taxel=rng.standard_normal(taxel_shape).astype(np.float32),
            )
        )
    return samples


def test_random_split_dataset_respects_val_ratio():
    dataset = TaxelDataset(_make_synthetic_samples(10, (2, 2), (1, 2)))

    train_subset, val_subset = random_split_dataset(dataset, val_ratio=0.2, seed=0)

    assert len(train_subset) == 8
    assert len(val_subset) == 2
    assert len(train_subset) + len(val_subset) == len(dataset)
    assert set(train_subset.indices).isdisjoint(set(val_subset.indices))


def test_random_split_dataset_is_reproducible_with_same_seed():
    dataset = TaxelDataset(_make_synthetic_samples(10, (2, 2), (1, 2)))

    train_a, val_a = random_split_dataset(dataset, val_ratio=0.3, seed=42)
    train_b, val_b = random_split_dataset(dataset, val_ratio=0.3, seed=42)

    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)
