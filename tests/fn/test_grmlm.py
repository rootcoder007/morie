"""Tests for grmlm.geron_bert_mlm_loss."""

from morie.fn import _array_core as np

from morie.fn.grmlm import geron_bert_mlm_loss


def test_grmlm_basic():
    """Test basic functionality."""
    logits = 0.5
    targets = 0.5
    mask = 1
    result = geron_bert_mlm_loss(logits, targets, mask)
    assert isinstance(result, dict)
    assert "estimate" in result or "loss" in result


def test_grmlm_edge():
    """Test edge cases."""
    logits = 0.5
    targets = 0.5
    mask = 1
    result = geron_bert_mlm_loss(logits, targets, mask)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmlm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
