"""Tests for dinopr.dino_self_distill."""

from morie.fn import _array_core as np

from morie.fn.dinopr import dino_self_distill


def test_dinopr_basic():
    """Test basic functionality."""
    s_logits = np.random.default_rng(42).normal(0, 1, 100)
    t_logits = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_self_distill(s_logits, t_logits)
    assert isinstance(result, dict)
    assert "loss" in result
def test_dinopr_edge():
    """Test edge cases."""
    s_logits = np.random.default_rng(42).normal(0, 1, 100)
    t_logits = np.random.default_rng(42).normal(0, 1, 100)
    result = dino_self_distill(s_logits, t_logits)
    assert isinstance(result, dict)
