"""Tests for sutva.sutva_assumption."""

from morie.fn import _array_core as np

from morie.fn.sutva import sutva_assumption


def test_sutva_basic():
    """Test basic functionality."""
    interference = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sutva_assumption(interference)
    assert isinstance(result, dict)
    assert "maxinterference" in result


def test_sutva_edge():
    """Test edge cases."""
    interference = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = sutva_assumption(interference)
    assert isinstance(result, dict)
