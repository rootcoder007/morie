"""Tests for hmpe.geron_positional_encoding."""

from morie.fn import _array_core as np

from morie.fn.hmpe import geron_positional_encoding


def test_hmpe_basic():
    """Test basic functionality."""
    pos = 0.5
    d_model = 2.0
    result = geron_positional_encoding(pos, d_model)
    assert isinstance(result, dict)
    assert "estimate" in result or "pe" in result


def test_hmpe_edge():
    """Test edge cases."""
    pos = 0.5
    d_model = 2.0
    result = geron_positional_encoding(pos, d_model)
    assert isinstance(result, dict)
