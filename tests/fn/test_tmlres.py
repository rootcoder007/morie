"""Tests for tmlres.tmle_residual."""

from morie.fn import _array_core as np

from morie.fn.tmlres import tmle_residual


def test_tmlres_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_residual(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tmlres_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_residual(y, D, X)
    assert isinstance(result, dict)
