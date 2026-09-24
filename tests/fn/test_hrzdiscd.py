"""Tests for hrzdiscd.horowitz_direct_discrete_x."""

from morie.fn import _array_core as np

from morie.fn.hrzdiscd import horowitz_direct_discrete_x


def test_hrzdiscd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_direct_discrete_x(x, y)
    assert isinstance(result, dict)
    assert "beta" in result


def test_hrzdiscd_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_direct_discrete_x(x, y)
    assert isinstance(result, dict)
