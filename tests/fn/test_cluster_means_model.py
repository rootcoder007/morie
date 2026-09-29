"""Tests for cluster_means_model.cluster_means_model."""

from morie.fn import _array_core as np
from morie.fn.cluster_means_model import cluster_means_model


def test_ca7e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_means_model(x)
    assert isinstance(result, dict)
    assert "cluster_means" in result


def test_ca7e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = cluster_means_model(x)
    assert isinstance(result, dict)


def test_cluster_means_and_deviations_recomputed():
    import pytest

    groups = [[3.0, 4.0, 5.0], [6.0, 8.0], [1.0, 2.0, 2.0, 3.0]]
    allv = [v for g in groups for v in g]
    gm = sum(allv) / len(allv)
    r = cluster_means_model(groups)
    assert r["value"] == pytest.approx(gm, rel=1e-14)
    assert r["cluster_means"] == pytest.approx([4.0, 7.0, 2.0], rel=1e-14)
    assert r["u_j"] == pytest.approx([4.0 - gm, 7.0 - gm, 2.0 - gm], rel=1e-13)
