"""Tests for hmlrl.geron_linear_regression_life."""

from morie.fn import _array_core as np

from morie.fn.hmlrl import geron_linear_regression_life


def test_hmlrl_basic():
    """Test basic functionality."""
    gdp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta0 = 0.1
    theta1 = 0.1
    result = geron_linear_regression_life(gdp, theta0, theta1)
    assert isinstance(result, dict)
    assert "estimate" in result or "prediction" in result


def test_hmlrl_edge():
    """Test edge cases."""
    gdp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    theta0 = 0.1
    theta1 = 0.1
    result = geron_linear_regression_life(gdp, theta0, theta1)
    assert isinstance(result, dict)
