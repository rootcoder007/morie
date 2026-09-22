"""Tests for acwhe.private_accuracy_tradeoff."""

from morie.fn import _array_core as np

from morie.fn.acwhe import private_accuracy_tradeoff


def test_acwhe_basic():
    """Test basic functionality."""
    result = private_accuracy_tradeoff()
    assert isinstance(result, dict)
    assert "noise_scale" in result
def test_acwhe_edge():
    """Test edge cases."""
    result = private_accuracy_tradeoff()
    assert isinstance(result, dict)
