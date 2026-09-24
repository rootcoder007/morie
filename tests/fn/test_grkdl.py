"""Tests for grkdl.geron_knowledge_distillation_loss."""

from morie.fn import _array_core as np

from morie.fn.grkdl import geron_knowledge_distillation_loss


def test_grkdl_basic():
    """Test basic functionality."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkdl_edge():
    """Test edge cases."""
    student_logits = np.random.default_rng(42).normal(0, 1, 100)
    teacher_logits = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_knowledge_distillation_loss(student_logits, teacher_logits, y, alpha, T)
    assert isinstance(result, dict)


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
