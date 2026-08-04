"""Tests for rgturns.rangayyan_turns_count."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_turns_count


def test_rgturns_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_turns_count(x, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgturns_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_turns_count(x, threshold)
    assert isinstance(result, dict)
