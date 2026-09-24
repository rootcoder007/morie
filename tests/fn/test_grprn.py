"""Tests for grprn.geron_weight_pruning."""

from morie.fn import _array_core as np

from morie.fn.grprn import geron_weight_pruning


def test_grprn_basic():
    """Test basic functionality."""
    W = [[1.0, -0.1], [0.05, 2.0]]
    sparsity = 0.5
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grprn_edge():
    """Test edge cases."""
    W = [[1.0, -0.1], [0.05, 2.0]]
    sparsity = 0.5
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grprn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
