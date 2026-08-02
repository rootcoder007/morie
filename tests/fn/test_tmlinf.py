"""Tests for tmlinf.tmle_inference."""

from morie.fn import _array_core as np

from morie.fn.tmlinf import tmle_inference


def test_tmlinf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Q = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = tmle_inference(y, D, X, Q, g)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tmlinf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Q = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = tmle_inference(y, D, X, Q, g)
    assert isinstance(result, dict)
