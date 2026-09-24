"""Tests for btwldcl.boot_wild_cluster."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.btwldcl import boot_wild_cluster


def test_btwldcl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    cluster = rng.integers(0, 5, n)
    result = boot_wild_cluster(X, y, cluster, B=20, seed=1, coef=1, beta0=0.0, alpha=0.05)
    assert isinstance(result, dict)
    assert "beta_hat" in result
    assert "se_cluster" in result
    assert "w" in result
    assert "p_value" in result
    assert "reject" in result
    assert "B" in result
    assert "G" in result
    assert "n" in result
    assert "p" in result
    assert len(result["beta_b"]) == 20
    assert len(result["w_b"]) == 20
    assert result["B"] == 20
    assert result["n"] == n
    assert result["p"] == p
    assert math.isfinite(result["se_cluster"])
    assert math.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert int(result["reject"]) in (0, 1)


def test_btwldcl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    cluster = rng.integers(0, 5, n)

    # alpha must lie strictly between 0 and 1
    with pytest.raises(ValueError):
        boot_wild_cluster(X, y, cluster, B=10, seed=1, coef=1, beta0=0.0, alpha=1.5)

    # B must be at least 2
    with pytest.raises(ValueError):
        boot_wild_cluster(X, y, cluster, B=1, seed=1, coef=1, beta0=0.0, alpha=0.05)

    # coef must be a valid index into the p columns
    with pytest.raises(ValueError):
        boot_wild_cluster(X, y, cluster, B=10, seed=1, coef=10, beta0=0.0, alpha=0.05)

    # X, y, and cluster must have the same length
    with pytest.raises(ValueError):
        boot_wild_cluster(X, y, cluster[: n - 1], B=10, seed=1, coef=1, beta0=0.0, alpha=0.05)

    # need more rows than columns: n <= p
    X_small = rng.normal(0, 1, (p, p))
    y_small = rng.normal(0, 1, p)
    cluster_small = rng.integers(0, 5, p)
    with pytest.raises(ValueError):
        boot_wild_cluster(X_small, y_small, cluster_small, B=10, seed=1, coef=1, beta0=0.0, alpha=0.05)
