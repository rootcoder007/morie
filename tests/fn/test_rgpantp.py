"""Tests for rgpantp.rangayyan_pan_tompkins."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_pan_tompkins


def test_rgpantp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_pan_tompkins(x)
    assert isinstance(result, dict)
    assert "qrs" in result


def test_rgpantp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_pan_tompkins(x)
    assert isinstance(result, dict)
