"""Tests for clcrp.clustered_crp."""

from morie.fn import _array_core as np

import pytest

from morie.fn.clcrp import clustered_crp


def test_clcrp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    y = rng.normal(0, 1, n)
    # Build a symmetric n x n distance matrix with zero diagonal
    raw = rng.normal(0, 1, (n, n))
    distances = np.abs(raw)
    for i in range(n):
        distances[i][i] = 0.0
        for j in range(i + 1, n):
            avg = (distances[i][j] + distances[j][i]) / 2.0
            distances[i][j] = avg
            distances[j][i] = avg

    result = clustered_crp(y, distances, alpha=0.5, decay=1.0, seed=42)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n" in result
    assert "n_clusters" in result
    assert "z" in result
    assert "links" in result
    assert "counts" in result
    assert "cluster_mean" in result
    assert result["n"] == n
    assert 1 <= result["n_clusters"] <= n
    assert len(result["z"]) == n
    assert len(result["links"]) == n


def test_clcrp_edge():
    """Test edge cases - empty y is invalid per the docstring."""
    with pytest.raises(ValueError):
        clustered_crp([], [[0.0]], alpha=1.0)
