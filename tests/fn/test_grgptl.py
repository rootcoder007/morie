"""Tests for grgptl.geron_gpt_autoregressive_loss."""

from morie.fn import _array_core as np

from morie.fn.grgptl import geron_gpt_autoregressive_loss


def test_grgptl_basic():
    """Test basic functionality."""
    logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    targets = [0, 2]
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgptl_edge():
    """Test edge cases."""
    logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    targets = [0, 2]
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grgptl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
