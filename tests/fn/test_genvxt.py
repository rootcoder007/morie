"""Tests for genvxt.generalizability_theory."""

from morie.fn import _array_core as np

from morie.fn.genvxt import generalizability_theory


def test_genvxt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    facets = np.random.default_rng(42).normal(0, 1, 100)
    result = generalizability_theory(X, facets)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_genvxt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    facets = np.random.default_rng(42).normal(0, 1, 100)
    result = generalizability_theory(X, facets)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.genvxt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
