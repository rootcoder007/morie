"""Tests for aitdrr.dirichlet_regression."""

import math

from morie.fn import _array_core as np

from morie.fn.aitdrr import dirichlet_regression


def _make_compositions(rng, n, D):
    """Build an n-by-D matrix where each row is strictly positive and sums to one."""
    raw = rng.uniform(0.1, 1.0, (n, D))
    rows = []
    for r in raw:
        s = sum(r)
        rows.append([v / s for v in r])
    return rows


def test_aitdrr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p, D = 40, 3, 4
    X_cov = rng.normal(0, 1, (n, p))
    Y_comp = _make_compositions(rng, n, D)
    result = dirichlet_regression(X_cov, Y_comp, ref=0)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "phi" in result
    assert "ll" in result
    assert "score_max_abs" in result
    assert len(result["beta"]) == p
    assert len(result["beta"][0]) == D
    assert math.isfinite(result["phi"])
    assert math.isfinite(result["ll"])
    assert result["score_max_abs"] >= 0


def test_aitdrr_edge():
    """Test edge cases with the minimum number of parts."""
    rng = np.random.default_rng(42)
    n, p, D = 40, 3, 2
    X_cov = rng.normal(0, 1, (n, p))
    Y_comp = _make_compositions(rng, n, D)
    result = dirichlet_regression(X_cov, Y_comp, ref=None)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "phi" in result
    assert "ll" in result
    assert "score_max_abs" in result
    assert len(result["beta"]) == p
    assert len(result["beta"][0]) == D
    assert math.isfinite(result["phi"])
    assert math.isfinite(result["ll"])
    assert result["score_max_abs"] >= 0
