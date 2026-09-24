"""Tests for mtsmtv.mts_mtr_combined."""

from morie.fn import _array_core as np

from morie.fn.mtsmtv import mts_mtr_combined


def test_mtsmtv_basic():
    """Test basic functionality."""
    y = 0.5
    D = 0.5
    y_min = 0.5
    y_max = 0.5
    result = mts_mtr_combined(y, D, y_min, y_max)
    assert isinstance(result, dict)
    assert "lower" in result


def test_mtsmtv_edge():
    """Test edge cases."""
    y = 0.5
    D = 0.5
    y_min = 0.5
    y_max = 0.5
    result = mts_mtr_combined(y, D, y_min, y_max)
    assert isinstance(result, dict)
