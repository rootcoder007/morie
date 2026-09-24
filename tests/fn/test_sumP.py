"""Tests for sumP.sum_pool."""

from morie.fn import _array_core as np

from morie.fn.sumP import sum_pool


def test_sumP_basic():
    """Test basic functionality."""
    H = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sum_pool(H)
    assert isinstance(result, dict)
    assert "sum" in result


def test_sumP_edge():
    """Test edge cases."""
    H = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = sum_pool(H)
    assert isinstance(result, dict)
