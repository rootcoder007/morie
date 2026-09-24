"""Tests for grinc.geron_in_context_learning."""

from morie.fn import _array_core as np

from morie.fn.grinc import geron_in_context_learning


def test_grinc_basic():
    """Test basic functionality."""
    examples = [('a', '1'), ('b', '2')]
    query = 'c'
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grinc_edge():
    """Test edge cases."""
    examples = [('a', '1'), ('b', '2')]
    query = 'c'
    result = geron_in_context_learning(examples, query)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grinc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
