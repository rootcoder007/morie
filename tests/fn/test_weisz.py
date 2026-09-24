"""Tests for weisz.weiszfeld."""

from morie.fn import _array_core as np

from morie.fn.weisz import weiszfeld


def test_weisz_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = weiszfeld(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_weisz_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = weiszfeld(X)
    assert isinstance(result, dict)
