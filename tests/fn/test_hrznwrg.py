"""Tests for hrznwrg.horowitz_nw_estimator_g."""

from morie.fn import _array_core as np

from morie.fn.hrznwrg import horowitz_nw_estimator_g


def test_hrznwrg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_nw_estimator_g(X, y, beta)
    assert isinstance(result, dict)
    assert "index_grid" in result


def test_hrznwrg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_nw_estimator_g(X, y, beta)
    assert isinstance(result, dict)
