"""Tests for alhds.alammar_hdbscan_cluster."""

from morie.fn.alhds import alammar_hdbscan_cluster


def test_alhds_basic():
    X = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1], [20, 20]]
    out = alammar_hdbscan_cluster(X, 3, 2)
    assert out["n_clusters"] == 2
    assert out["labels"][6] == -1


def test_alhds_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        alammar_hdbscan_cluster([[0, 0], [1, 1], [2, 2]], 1)
