"""Tests for cvxind.boyd_indicator."""

from morie.fn import _array_core as np

from morie.fn.cvxind import boyd_indicator


def test_cvxind_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_indicator(x)
    assert isinstance(result, dict)
    assert "value" in result
def test_cvxind_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_indicator(x)
    assert isinstance(result, dict)
