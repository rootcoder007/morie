"""Tests for kmchin.kamath_chinchilla_compute_optimal."""

from morie.fn import _array_core as np

from morie.fn.kmchin import kamath_chinchilla_compute_optimal


def test_kmchin_basic():
    """Test basic functionality."""
    compute_budget = 12000000000.0
    result = kamath_chinchilla_compute_optimal(compute_budget)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmchin_edge():
    """Test edge cases."""
    compute_budget = 12000000000.0
    result = kamath_chinchilla_compute_optimal(compute_budget)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmchin as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
