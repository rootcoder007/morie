"""Tests for wsmvit.wasserman_viterbi."""

from morie.fn import _array_core as np

from morie.fn.wsmvit import wasserman_viterbi


def test_wsmvit_basic():
    """Test basic functionality."""
    obs = 0.5
    A = 0.5
    B = 0.5
    pi = 0.5
    result = wasserman_viterbi(obs, A, B, pi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmvit_edge():
    """Test edge cases."""
    obs = 0.5
    A = 0.5
    B = 0.5
    pi = 0.5
    result = wasserman_viterbi(obs, A, B, pi)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmvit as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
