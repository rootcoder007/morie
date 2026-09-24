"""Tests for hmkd.geron_knowledge_distillation."""

import math

from morie.fn import _array_core as np

from morie.fn.hmkd import geron_knowledge_distillation


def test_hmkd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m, C = 40, 3
    teacher = rng.normal(0.0, 1.0, (m, C))
    student = rng.normal(0.0, 1.0, (m, C))
    y = rng.integers(0, C, m)
    result = geron_knowledge_distillation(teacher, student, y=y, T=2.0, alpha=0.5)
    assert isinstance(result, dict)
    for key in ("loss", "ce_loss", "kl_loss", "teacher_probs", "student_probs",
                "agreement", "estimate", "n", "method"):
        assert key in result
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["ce_loss"])
    assert math.isfinite(result["kl_loss"])
    assert result["n"] == m
    assert len(result["teacher_probs"]) == m
    assert len(result["student_probs"]) == m
    assert len(result["teacher_probs"][0]) == C


def test_hmkd_edge():
    """Test edge cases: pure distillation with alpha=0 needs no labels."""
    rng = np.random.default_rng(42)
    m, C = 40, 3
    teacher = rng.normal(0.0, 1.0, (m, C))
    student = rng.normal(0.0, 1.0, (m, C))
    result = geron_knowledge_distillation(teacher, student, T=1.0, alpha=0.0)
    assert isinstance(result, dict)
    assert math.isfinite(result["loss"])
    # alpha=0 means pure distillation: loss equals kl_loss exactly
    assert abs(result["loss"] - result["kl_loss"]) < 1e-12


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmkd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
