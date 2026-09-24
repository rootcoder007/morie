"""Tests for kmcot.kamath_chain_of_thought."""

from morie.fn import _array_core as np

from morie.fn.kmcot import kamath_chain_of_thought


def test_kmcot_basic():
    """Test basic functionality."""
    prompt = '2+2?'
    model = lambda p: 'add. Answer: 4'
    result = kamath_chain_of_thought(prompt, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcot_edge():
    """Test edge cases."""
    prompt = '2+2?'
    model = lambda p: 'add. Answer: 4'
    result = kamath_chain_of_thought(prompt, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcot as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
