"""Tests for fzbfkde2.fauzi_bdfree_density_from_cdf."""

from morie.fn import _array_core as np

from morie.fn.fzbfkde2 import fauzi_bdfree_density_from_cdf


def test_fzbfkde2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = lambda v: v
    dg_func = lambda v: 1
    result = fauzi_bdfree_density_from_cdf(x, bandwidth, g_func, dg_func)
    assert isinstance(result, dict)
    assert "estimate" in result or "density" in result


def test_fzbfkde2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    bandwidth = 0.5
    g_func = lambda v: v
    dg_func = lambda v: 1
    result = fauzi_bdfree_density_from_cdf(x, bandwidth, g_func, dg_func)
    assert isinstance(result, dict)
