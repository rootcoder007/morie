"""Tests for hmdbrt.geron_distilbert."""

from morie.fn import _array_core as np

from morie.fn.hmdbrt import geron_distilbert


def test_hmdbrt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    B, C = 8, 4
    teacher = rng.normal(0, 1, (B, C))
    student = rng.normal(0, 1, (B, C))
    X = rng.normal(0, 1, (B, 3))
    mlm_labels = rng.integers(0, C, B)
    result = geron_distilbert(teacher, student, X, mlm_labels=mlm_labels)
    assert isinstance(result, dict)
    for key in ("loss", "loss_ce", "loss_mlm", "loss_cos",
                "teacher_params", "student_params", "param_reduction",
                "agreement", "estimate", "n", "method"):
        assert key in result
    assert result["n"] == B


def test_hmdbrt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    B, C = 4, 3
    teacher = rng.normal(0, 1, (B, C))
    student = rng.normal(0, 1, (B, C))
    X = rng.normal(0, 1, (B, 2))
    # alpha_mlm=0 so mlm_labels is not required.
    result = geron_distilbert(teacher, student, X, alpha_mlm=0.0)
    assert isinstance(result, dict)
    assert "loss" in result
    assert "agreement" in result
    assert "method" in result


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdbrt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
