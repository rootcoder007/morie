"""Tests for kmtot.kamath_tree_of_thoughts."""

from morie.fn import _array_core as np

from morie.fn.kmtot import kamath_tree_of_thoughts


def test_kmtot_basic():
    """Test basic functionality."""
    problem = np.random.default_rng(42).normal(0, 1, 100)
    branch_factor = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_tree_of_thoughts(problem, branch_factor, max_depth, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmtot_edge():
    """Test edge cases."""
    problem = np.random.default_rng(42).normal(0, 1, 100)
    branch_factor = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_tree_of_thoughts(problem, branch_factor, max_depth, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtot as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
