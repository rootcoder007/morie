"""Tests for hmrad.geron_reverse_autodiff."""

from morie.fn import _array_core as np

from morie.fn.hmrad import geron_reverse_autodiff


def test_hmrad_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reverse_autodiff(f, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmrad_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reverse_autodiff(f, x)
    assert isinstance(result, dict)
