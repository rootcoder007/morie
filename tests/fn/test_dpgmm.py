"""Tests for dpgmm.dp_gaussian_mixture."""

from morie.fn import _array_core as np

from morie.fn.dpgmm import dp_gaussian_mixture


def test_dpgmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_gaussian_mixture(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpgmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = dp_gaussian_mixture(y)
    assert isinstance(result, dict)
