"""Tests for t_paired.t_paired."""

from morie.fn import _array_core as np

from morie.fn.t_paired import t_paired


def test_ca9e10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = t_paired(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca9e10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = t_paired(x)
    assert isinstance(result, dict)
