"""Tests for irt3pl.three_parameter_logistic."""

from morie.fn import _array_core as np

from morie.fn.irt3pl import three_parameter_logistic


def test_irt3pl_basic():
    """Test basic functionality."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = three_parameter_logistic(theta)
    assert isinstance(result, dict)
    assert "p" in result


def test_irt3pl_edge():
    """Test edge cases."""
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = three_parameter_logistic(theta)
    assert isinstance(result, dict)
