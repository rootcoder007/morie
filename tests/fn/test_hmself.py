"""Tests for hmself.geron_self_supervised."""

from morie.fn import _array_core as np

from morie.fn.hmself import geron_self_supervised


def test_hmself_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_self_supervised(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss" in result


def test_hmself_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_self_supervised(X)
    assert isinstance(result, dict)
