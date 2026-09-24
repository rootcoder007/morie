"""Tests for hmdrp.geron_dropout."""

from morie.fn import _array_core as np

from morie.fn.hmdrp import geron_dropout


def test_hmdrp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = geron_dropout(x, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "y" in result


def test_hmdrp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    p = 0.1
    result = geron_dropout(x, p)
    assert isinstance(result, dict)
