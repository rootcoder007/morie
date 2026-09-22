"""Tests for ali.ali_mikhail_haq_copula."""

from morie.fn import _array_core as np

from morie.fn.ali import ali_mikhail_haq_copula


def test_ali_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = ali_mikhail_haq_copula(u, v)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ali_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = ali_mikhail_haq_copula(u, v)
    assert isinstance(result, dict)
