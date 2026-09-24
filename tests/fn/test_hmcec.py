"""Tests for hmcec.geron_cross_entropy_cost."""

from morie.fn import _array_core as np

from morie.fn.hmcec import geron_cross_entropy_cost


def test_hmcec_basic():
    """Test basic functionality."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
    assert "estimate" in result or "cost" in result


def test_hmcec_edge():
    """Test edge cases."""
    X = 5
    Y = 5
    theta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmcec as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
