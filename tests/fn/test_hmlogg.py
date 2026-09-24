"""Tests for hmlogg.geron_logistic_gradient."""

from morie.fn import _array_core as np

from morie.fn.hmlogg import geron_logistic_gradient


def test_hmlogg_basic():
    """Test basic functionality."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "gradient" in result


def test_hmlogg_edge():
    """Test edge cases."""
    X = 0.5
    y = 1
    theta = 0.5
    result = geron_logistic_gradient(X, y, theta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmlogg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
