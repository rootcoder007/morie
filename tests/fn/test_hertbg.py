"""Tests for hertbg.heritability."""
import math

from morie.fn import _array_core as np

from morie.fn.hertbg import heritability


def test_hertbg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 40
    p = 100
    y = rng.normal(0, 1, n)
    # Build a valid positive definite kinship matrix K = X X^T / p
    X = rng.normal(0, 1, (n, p))
    K = X @ X.T / p + np.eye(n) * 0.01
    result = heritability(y, K)
    assert isinstance(result, dict)
    assert "estimate" in result
    est = float(result["estimate"])
    assert math.isfinite(est)
    assert 0.0 <= est <= 1.0
    assert "h2" in result
    assert math.isfinite(float(result["h2"]))
    assert "n" in result and int(result["n"]) == n


def test_hertbg_edge():
    """Test edge case with small valid input."""
    rng = np.random.default_rng(43)
    n = 6
    p = 20
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, p))
    K = X @ X.T / p + np.eye(n) * 0.01
    result = heritability(y, K, grid=5, refine=3)
    assert isinstance(result, dict)
    assert "estimate" in result
    est = float(result["estimate"])
    assert math.isfinite(est)
    assert 0.0 <= est <= 1.0
    assert "grid_h2" in result and "grid_loglik" in result
    assert len(result["grid_h2"]) == 5
    assert len(result["grid_loglik"]) == 5
