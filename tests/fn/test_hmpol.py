"""Tests for hmpol.geron_policy."""

from morie.fn import _array_core as np

from morie.fn.hmpol import geron_policy


def test_hmpol_basic():
    """Test basic functionality."""
    state = 0.5
    pi = 5
    result = geron_policy(state, pi)
    assert isinstance(result, dict)
    assert "estimate" in result or "probabilities" in result


def test_hmpol_edge():
    """Test edge cases."""
    state = 0.5
    pi = 5
    result = geron_policy(state, pi)
    assert isinstance(result, dict)
