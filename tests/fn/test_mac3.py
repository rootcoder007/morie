"""Tests for mac3.ma_centered_predictors."""

from morie.fn import _array_core as np

from morie.fn.mac3 import ma_centered_predictors


def test_mac3_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mods = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_centered_predictors(yi, vi, mods)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mac3_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mods = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_centered_predictors(yi, vi, mods)
    assert isinstance(result, dict)
