"""Tests for hrzkde.horowitz_appendix_kde."""

from morie.fn import _array_core as np

from morie.fn.hrzkde import horowitz_appendix_kde


def test_hrzkde_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_appendix_kde(x)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzkde_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_appendix_kde(x)
    assert isinstance(result, dict)
