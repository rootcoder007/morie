"""Tests for marpct.ma_percent_heterogeneity_R2."""

from morie.fn import _array_core as np

from morie.fn.marpct import ma_percent_heterogeneity_R2


def test_marpct_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mods = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_percent_heterogeneity_R2(yi, vi, mods)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_marpct_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    mods = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_percent_heterogeneity_R2(yi, vi, mods)
    assert isinstance(result, dict)
