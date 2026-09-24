"""Tests for potM.peaks_over_threshold."""

from morie.fn import _array_core as np

from morie.fn.potM import peaks_over_threshold


def test_potM_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    u = 0.1
    result = peaks_over_threshold(y, u)
    assert isinstance(result, dict)
    assert "sigma" in result


def test_potM_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    u = 0.1
    result = peaks_over_threshold(y, u)
    assert isinstance(result, dict)
