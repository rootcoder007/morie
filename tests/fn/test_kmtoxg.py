"""Tests for kmtoxg.kamath_toxigen_score."""

import math

from morie.fn import _array_core as np

from morie.fn.kmtoxg import kamath_toxigen_score


def test_kmtoxg_basic():
    """Test basic functionality with a callable classifier returning a scalar probability."""
    text = "hello world"
    classifier = lambda t: 0.1
    result = kamath_toxigen_score(text, classifier)
    assert isinstance(result, dict)
    assert math.isclose(result["estimate"], 0.1)
    assert math.isclose(result["probability"], 0.1)
    assert result["toxic"] is False
    assert math.isclose(result["threshold"], 0.5)
    assert result["text"] == text
    assert result["n"] == 1


def test_kmtoxg_edge():
    """Test edge cases with a callable returning a (p_benign, p_toxic) pair."""
    text = "some text"
    classifier = lambda t: (0.2, 0.8)
    result = kamath_toxigen_score(text, classifier)
    assert isinstance(result, dict)
    assert math.isclose(result["estimate"], 0.8)
    assert result["toxic"] is True


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtoxg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
