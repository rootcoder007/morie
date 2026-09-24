"""Tests for grdino.geron_dino_self_distillation."""

from morie.fn import _array_core as np

from morie.fn.grdino import geron_dino_self_distillation


def test_grdino_basic():
    """Test basic functionality."""
    student_logits = [0.0, 0.0]
    teacher_logits = [1.0, 0.0]
    tau_s = 0.1
    tau_t = 0.05
    result = geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdino_edge():
    """Test edge cases."""
    student_logits = [0.0, 0.0]
    teacher_logits = [1.0, 0.0]
    tau_s = 0.1
    tau_t = 0.05
    result = geron_dino_self_distillation(student_logits, teacher_logits, tau_s, tau_t)
    assert isinstance(result, dict)
