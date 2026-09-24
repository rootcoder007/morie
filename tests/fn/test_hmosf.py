"""Tests for hmosf.geron_one_shot."""

from morie.fn import _array_core as np

from morie.fn.hmosf import geron_one_shot


def test_hmosf_basic():
    """Test basic functionality."""
    copy = lambda prompt: prompt[0][1]
    example = ("hello", "greeting")
    query = "goodbye"
    result = geron_one_shot(copy, example, query)
    assert "prediction" in result
    assert result["prediction"] == "greeting"
    assert result["shots"] == 1
    assert result["prompt"] == [("hello", "greeting"), ("goodbye", None)]
    assert result["demo_label"] == "greeting"


def test_hmosf_edge():
    """Test edge cases."""
    rule = lambda p: "greeting" if "hello" in p[-1][0] else "farewell"
    example = ("hello there", "greeting")
    query = "goodbye now"
    result = geron_one_shot(rule, example, query)
    assert "prediction" in result
    assert result["prediction"] == "farewell"


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmosf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
