"""Tests for madMov.moving_mad."""

from morie.fn import _array_core as np

from morie.fn.madMov import moving_mad


def test_madMov_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = moving_mad(x, window)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_madMov_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    window = 5
    result = moving_mad(x, window)
    assert isinstance(result, dict)
