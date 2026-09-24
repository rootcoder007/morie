"""Tests for grkdl.geron_knowledge_distillation_loss."""

import math

from morie.fn import _array_core as np

from morie.fn.grkdl import geron_knowledge_distillation_loss


def test_grkdl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m, K = 40, 3
    student_logits = rng.normal(0, 1, (m, K))
    teacher_logits = rng.normal(0, 1, (m, K))
    y = rng.integers(0, K, m)
    result = geron_knowledge_distillation_loss(
        student_logits, teacher_logits, y, alpha=0.5, T=2.0
    )
    assert isinstance(result, dict)
    for key in ("loss", "ce_hard", "kl_soft", "kl_student_teacher",
                "kl_teacher_student", "soft_targets", "teacher_entropy",
                "estimate", "n", "method"):
        assert key in result
    for key in ("loss", "ce_hard", "kl_soft", "kl_student_teacher",
                "kl_teacher_student", "teacher_entropy"):
        assert math.isfinite(result[key])
    assert result["n"] == m
    assert len(result["soft_targets"]) == m


def test_grkdl_edge():
    """Test edge cases with boundary alpha."""
    rng = np.random.default_rng(7)
    m, K = 10, 4
    student_logits = rng.normal(0, 1, (m, K))
    teacher_logits = rng.normal(0, 1, (m, K))
    y = rng.integers(0, K, m)
    result = geron_knowledge_distillation_loss(
        student_logits, teacher_logits, y, alpha=0.0, T=2.0
    )
    assert isinstance(result, dict)
    for key in ("loss", "ce_hard", "kl_soft", "kl_student_teacher",
                "kl_teacher_student", "soft_targets", "teacher_entropy",
                "estimate", "n", "method"):
        assert key in result
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["ce_hard"])
    assert math.isfinite(result["kl_soft"])
    assert math.isfinite(result["teacher_entropy"])
    assert result["n"] == m
    assert len(result["soft_targets"]) == m


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grkdl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
