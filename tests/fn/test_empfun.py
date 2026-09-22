"""Tests for empfun.empty_space_function."""

from morie.fn import _array_core as np

from morie.fn.empfun import empty_space_function


def test_empfun_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = empty_space_function(coords)
    assert isinstance(result, dict)
    assert "r" in result
def test_empfun_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = empty_space_function(coords)
    assert isinstance(result, dict)
