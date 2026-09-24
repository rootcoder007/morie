"""Tests for hmmnl.geron_multinomial_logistic."""

from morie.fn import _array_core as np

from morie.fn.hmmnl import geron_multinomial_logistic


def test_hmmnl_basic():
    """Test basic functionality."""
    X = 0.5
    Y = 5
    result = geron_multinomial_logistic(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "Theta" in result


def test_hmmnl_edge():
    """Test edge cases."""
    X = 0.5
    Y = 5
    result = geron_multinomial_logistic(X, Y)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmnl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
