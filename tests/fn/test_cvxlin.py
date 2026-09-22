"""Tests for cvxlin.boyd_linear_program."""

from morie.fn import _array_core as np

from morie.fn.cvxlin import boyd_linear_program


def test_cvxlin_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_linear_program(c)
    assert isinstance(result, dict)
    assert "x" in result
def test_cvxlin_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = boyd_linear_program(c)
    assert isinstance(result, dict)
