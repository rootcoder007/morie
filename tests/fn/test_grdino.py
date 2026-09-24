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


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grdino as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
