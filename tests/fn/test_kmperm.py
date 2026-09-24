"""Tests for kmperm.kamath_permutation_lm_loss."""

from morie.fn import _array_core as np

from morie.fn.kmperm import kamath_permutation_lm_loss


def test_kmperm_basic():
    """Test basic functionality."""
    logits = 0.5
    targets = 0.5
    permutation = 0.5
    result = kamath_permutation_lm_loss(logits, targets, permutation)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss" in result


def test_kmperm_edge():
    """Test edge cases."""
    logits = 0.5
    targets = 0.5
    permutation = 0.5
    result = kamath_permutation_lm_loss(logits, targets, permutation)
    assert isinstance(result, dict)
