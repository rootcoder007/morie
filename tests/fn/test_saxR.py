"""Tests for saxR.sax_representation."""

from morie.fn import _array_core as np

from morie.fn.saxR import sax_representation


def test_saxR_basic():
    """Test basic functionality."""
    x = 0.5
    window = 1
    alphabet = 5
    result = sax_representation(x, window, alphabet)
    assert isinstance(result, dict)
    assert "estimate" in result or "word" in result


def test_saxR_edge():
    """Test edge cases."""
    x = 0.5
    window = 1
    alphabet = 5
    result = sax_representation(x, window, alphabet)
    assert isinstance(result, dict)
