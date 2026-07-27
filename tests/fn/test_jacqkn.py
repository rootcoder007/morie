"""Tests for jacqkn.jacquez_k_nn_test (Jacquez 1996)."""

import numpy as np
import pytest

from morie.fn.jacqkn import _knn_indicator, jacquez_k_nn_test


def test_knn_indicator_picks_exactly_k_and_excludes_self():
    D = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(float)
    A = _knn_indicator(D, 2)
    assert A.sum(axis=1).tolist() == [2] * 6
    assert not A.diagonal().any()


def test_no_interaction_is_not_detected():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 1, (60, 2))
    time = rng.uniform(0, 10, 60)
    assert jacquez_k_nn_test(coords, time, k=3, B=199, seed=2)["p_value"] > 0.05


def test_space_time_clustering_is_detected():
    """Cases in the same cluster share both a place and a moment."""
    rng = np.random.default_rng(3)
    centres = rng.uniform(0, 1, (6, 2))
    times = rng.uniform(0, 10, 6)
    idx = np.repeat(np.arange(6), 12)
    coords = centres[idx] + rng.normal(0, 0.01, (72, 2))
    time = times[idx] + rng.normal(0, 0.05, 72)
    assert jacquez_k_nn_test(coords, time, k=3, B=199, seed=2)["p_value"] <= 0.01


def test_clustering_in_space_alone_is_not_interaction():
    """Tight spatial clusters with times shuffled across them are null."""
    rng = np.random.default_rng(4)
    centres = rng.uniform(0, 1, (6, 2))
    idx = np.repeat(np.arange(6), 12)
    coords = centres[idx] + rng.normal(0, 0.01, (72, 2))
    time = rng.uniform(0, 10, 72)
    assert jacquez_k_nn_test(coords, time, k=3, B=199, seed=2)["p_value"] > 0.05


def test_statistic_is_bounded_by_the_number_of_ordered_pairs():
    rng = np.random.default_rng(5)
    res = jacquez_k_nn_test(rng.uniform(0, 1, (40, 2)), rng.uniform(0, 10, 40), k=4, B=19, seed=1)
    assert 0 <= res["statistic"] <= 40 * 4


def test_identical_space_and_time_orderings_saturate_the_count():
    """When time order mirrors position on a line, every space
    neighbour is also a time neighbour, so J_k hits its maximum."""
    x = np.arange(30, dtype=float).reshape(-1, 1)
    res = jacquez_k_nn_test(x, x.ravel(), k=2, B=19, seed=1)
    assert res["statistic"] == 30 * 2


def test_seed_makes_it_reproducible():
    rng = np.random.default_rng(6)
    c, t = rng.uniform(0, 1, (30, 2)), rng.uniform(0, 10, 30)
    assert jacquez_k_nn_test(c, t, k=3, B=49, seed=9)["p_value"] == jacquez_k_nn_test(c, t, k=3, B=49, seed=9)["p_value"]


def test_validates_inputs():
    rng = np.random.default_rng(7)
    c, t = rng.uniform(0, 1, (20, 2)), rng.uniform(0, 10, 20)
    with pytest.raises(ValueError, match="one entry per case"):
        jacquez_k_nn_test(c, t[:-1])
    with pytest.raises(ValueError, match="k must be at least 1"):
        jacquez_k_nn_test(c, t, k=0)
    with pytest.raises(ValueError, match="smaller than the number of cases"):
        jacquez_k_nn_test(c, t, k=20)
    with pytest.raises(ValueError, match="must be finite"):
        jacquez_k_nn_test(c, np.concatenate([t[:-1], [np.nan]]))
    with pytest.raises(ValueError, match="B must be at least 1"):
        jacquez_k_nn_test(c, t, B=0)
