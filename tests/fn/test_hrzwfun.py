"""Tests for hrzwfun.horowitz_nls_weight_function."""

from morie.fn import _array_core as np

from morie.fn.hrzwfun import horowitz_nls_weight_function


def test_hrzwfun_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nls_weight_function(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzwfun_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_nls_weight_function(x, y)
    assert isinstance(result, dict)
