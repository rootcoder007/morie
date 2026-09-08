"""Tests for clrgrf. Full anchor: ledger/wave3/anchor_grf_family.py."""
import pytest
from morie.fn.clrgrf import cluster_forest, cluster_index
from ._grf_fixture import clustered


@pytest.fixture(scope="module")
def d():
    return clustered(20, 15, 51)


def test_cluster_index_groups_in_first_seen_order():
    groups, labels = cluster_index(["a", "b", "a", "c"])
    assert groups == [[0, 2], [1], [3]]
    assert labels == ["a", "b", "c"]


def test_cluster_sampling_widens_the_interval(d):
    """Row-level sampling splits clusters across the split and estimate
    halves, violating honesty THROUGH the cluster while the interval
    still looks respectable."""
    cl = cluster_forest(d["y"], d["X"], d["cluster"], at=[[0.0, 0.0]],
                        n_trees=100, min_leaf=5, seed=7)
    row = cluster_forest(d["y"], d["X"], d["cluster"], at=[[0.0, 0.0]],
                         n_trees=100, min_leaf=5, seed=7,
                         cluster_sampling=False)
    assert cl["se"][0] > row["se"][0]
    assert cl["n_clusters"] == 20
    assert sum(cl["cluster_sizes"]) == d["n"]


def test_argument_checks(d):
    with pytest.raises(ValueError):
        cluster_forest(d["y"], d["X"], d["cluster"], unit="nope")
    with pytest.raises(ValueError):
        cluster_forest(d["y"][:30], d["X"][:30], ["a"] * 15 + ["b"] * 15)
    with pytest.raises(ValueError):
        cluster_forest(d["y"], d["X"], d["cluster"][:-1])
