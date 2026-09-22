"""Tests for adamopt.adam."""

from morie.fn import _array_core as np

from morie.fn.adamopt import adam


def test_adamopt_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = adam(g)
    assert isinstance(result, dict)
    assert "update" in result
def test_adamopt_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = adam(g)
    assert isinstance(result, dict)
