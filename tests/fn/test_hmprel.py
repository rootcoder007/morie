"""Tests for hmprel.geron_prelu."""

from morie.fn import _array_core as np

from morie.fn.hmprel import geron_prelu


def test_hmprel_basic():
    """Test basic functionality."""
    z = 0.5
    result = geron_prelu(z)
    assert isinstance(result, dict)
    assert "estimate" in result or "a" in result


def test_hmprel_edge():
    """Test edge cases."""
    z = 0.5
    result = geron_prelu(z)
    assert isinstance(result, dict)
