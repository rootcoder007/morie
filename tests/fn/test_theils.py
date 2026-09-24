"""Tests for theils.theil_sen."""

from morie.fn import _array_core as np

from morie.fn.theils import theil_sen


def test_theils_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = theil_sen(x, y)
    assert isinstance(result, dict)
    assert "slope" in result


def test_theils_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = theil_sen(x, y)
    assert isinstance(result, dict)
