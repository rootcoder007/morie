"""Tests for aithil.compositional_hill."""

from morie.fn import _array_core as np

from morie.fn.aithil import compositional_hill


def test_aithil_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = compositional_hill(x)
    assert isinstance(result, dict)
    assert "hill" in result
def test_aithil_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = compositional_hill(x)
    assert isinstance(result, dict)
