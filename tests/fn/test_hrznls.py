"""Tests for hrznls.horowitz_nls_sim."""

from morie.fn import _array_core as np

from morie.fn.hrznls import horowitz_nls_sim


def test_hrznls_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nls_sim(X, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrznls_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nls_sim(X, y)
    assert isinstance(result, dict)
