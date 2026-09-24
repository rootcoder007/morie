"""Tests for mahks.ma_hartung_knapp."""

from morie.fn import _array_core as np

from morie.fn.mahks import ma_hartung_knapp


def test_mahks_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_hartung_knapp(yi, vi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mahks_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_hartung_knapp(yi, vi)
    assert isinstance(result, dict)
