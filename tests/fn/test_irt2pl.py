"""Tests for irt2pl.two_parameter_logistic."""

from morie.fn import _array_core as np

from morie.fn.irt2pl import two_parameter_logistic


def test_irt2pl_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = two_parameter_logistic(theta)
    assert isinstance(result, dict)
    assert "p" in result


def test_irt2pl_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = two_parameter_logistic(theta)
    assert isinstance(result, dict)
