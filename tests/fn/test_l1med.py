"""Tests for l1med.l1_median."""

from morie.fn import _array_core as np
from morie.fn.l1med import l1_median


def test_l1med_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = l1_median(X, tol)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_l1med_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tol = 1e-6
    result = l1_median(X, tol)
    assert isinstance(result, dict)


def test_weiszfeld_iterates_replayed():
    import math

    import pytest

    X = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [1.0, 1.0], [5.0, 5.0]]
    mu = [sum(r[j] for r in X) / 5 for j in range(2)]
    for _ in range(50):
        w = [1 / math.dist(r, mu) for r in X]
        mu = [sum(wi * r[j] for wi, r in zip(w, X)) / sum(w) for j in range(2)]
    r = l1_median(X, max_iter=50)
    assert r["estimate"] == pytest.approx(mu, rel=1e-12)
    assert r["cost"] == pytest.approx(sum(math.dist(row, mu) for row in X), rel=1e-12)
    sr = [sum((row[j] - mu[j]) / math.dist(row, mu) for row in X) for j in range(2)]
    assert r["spatial_rank_norm"] == pytest.approx(math.hypot(*sr), rel=1e-9, abs=1e-12)
