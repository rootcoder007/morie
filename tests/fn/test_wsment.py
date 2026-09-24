"""Tests for wsment.wasserman_entropy."""

from morie.fn.wsment import wasserman_entropy


def test_wsment_basic():
    """Test basic functionality."""
    p = 5
    result = wasserman_entropy(p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsment_edge():
    """Test edge cases."""
    p = 5
    result = wasserman_entropy(p)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsment as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
