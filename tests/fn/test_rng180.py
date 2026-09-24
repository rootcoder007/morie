"""Tests for rng180.rangayyan_ch4_qrs_smoothing_ma_filter."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_qrs_smoothing_ma_filter


def test_rng180_basic():
    """Test basic functionality."""
    g1 = 0.1
    result = rangayyan_ch4_qrs_smoothing_ma_filter(g1)
    assert isinstance(result, dict)
    assert "g" in result


def test_rng180_edge():
    """Test edge cases."""
    g1 = 0.1
    result = rangayyan_ch4_qrs_smoothing_ma_filter(g1)
    assert isinstance(result, dict)
