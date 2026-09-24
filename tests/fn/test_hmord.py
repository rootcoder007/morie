"""Tests for hmord.geron_ordinal_encoding."""

from morie.fn import _array_core as np

from morie.fn.hmord import geron_ordinal_encoding


def test_hmord_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_ordinal_encoding(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "encoded" in result


def test_hmord_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_ordinal_encoding(X)
    assert isinstance(result, dict)
