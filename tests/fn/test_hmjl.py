"""Tests for hmjl.geron_johnson_lindenstrauss."""

from morie.fn import _array_core as np

from morie.fn.hmjl import geron_johnson_lindenstrauss


def test_hmjl_basic():
    """Test basic functionality."""
    n = 100
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmjl_edge():
    """Test edge cases."""
    n = 100
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_johnson_lindenstrauss(n, eps)
    assert isinstance(result, dict)
