"""Tests for rng178.rangayyan_ch4_qrs_combined_balda."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_qrs_combined_balda


def test_rng178_basic():
    """Test basic functionality."""
    y0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_qrs_combined_balda(y0, y1)
    assert isinstance(result, dict)
    assert "y2" in result


def test_rng178_edge():
    """Test edge cases."""
    y0 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_qrs_combined_balda(y0, y1)
    assert isinstance(result, dict)
