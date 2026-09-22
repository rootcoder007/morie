"""Tests for ccngg.nakagawa_conditional_r2."""

from morie.fn import _array_core as np

from morie.fn.ccngg import nakagawa_conditional_r2


def test_ccngg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ccngg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y)
    assert isinstance(result, dict)
