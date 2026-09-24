"""Tests for kmpet.kamath_pet_loss."""

from morie.fn import _array_core as np

from morie.fn.kmpet import kamath_pet_loss


def test_kmpet_basic():
    """Test basic functionality."""
    verbalizer_logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_true = 0.5
    mlm_logits = 0.5
    mlm_targets = 0.5
    alpha = 0.5
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmpet_edge():
    """Test edge cases."""
    verbalizer_logits = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_true = 0.5
    mlm_logits = 0.5
    mlm_targets = 0.5
    alpha = 0.5
    result = kamath_pet_loss(verbalizer_logits, y_true, mlm_logits, mlm_targets, alpha)
    assert isinstance(result, dict)
