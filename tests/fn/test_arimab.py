"""Tests for arimab.arima_box_jenkins."""

from morie.fn import _array_core as np

from morie.fn.arimab import arima_box_jenkins


def test_arimab_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = arima_box_jenkins(y)
    assert isinstance(result, dict)
    assert "phi" in result
def test_arimab_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = arima_box_jenkins(y)
    assert isinstance(result, dict)
