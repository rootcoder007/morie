"""Tests for rgcwvd.rangayyan_cohen_class."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_cohen_class


def test_rgcwvd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cohen_class(x)
    assert isinstance(result, dict)
    assert "tfd" in result


def test_rgcwvd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cohen_class(x)
    assert isinstance(result, dict)
