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


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmperm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
