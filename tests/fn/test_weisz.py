"""Tests for weisz.weiszfeld."""

from morie.fn import _array_core as np
from morie.fn.weisz import weiszfeld


def test_weisz_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = weiszfeld(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_weisz_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = weiszfeld(X)
    assert isinstance(result, dict)


def test_geometric_median_iterates_replayed():
    import math

    import pytest

    X = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [1.0, 1.0], [5.0, 5.0], [2.0, -1.0]]
    mu = [sum(r[j] for r in X) / 6 for j in range(2)]
    for _ in range(80):
        w = [1 / math.dist(r, mu) for r in X]
        mu = [sum(wi * r[j] for wi, r in zip(w, X)) / sum(w) for j in range(2)]
    r = weiszfeld(X, max_iter=80)
    assert r["estimate"] == pytest.approx(mu, rel=1e-12)
    assert r["cost"] == pytest.approx(sum(math.dist(p, mu) for p in X), rel=1e-12)
