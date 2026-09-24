"""Tests for wsmhmm.wasserman_hmm_forward."""

from morie.fn import _array_core as np

from morie.fn.wsmhmm import wasserman_hmm_forward


def test_wsmhmm_basic():
    """Test basic functionality."""
    obs = 0.5
    A = 1
    B = 1
    pi = 1
    result = wasserman_hmm_forward(obs, A, B, pi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmhmm_edge():
    """Test edge cases."""
    obs = 0.5
    A = 1
    B = 1
    pi = 1
    result = wasserman_hmm_forward(obs, A, B, pi)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmhmm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
