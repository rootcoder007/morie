"""Tests for kmstb.kamath_step_back_prompting."""

from morie.fn import _array_core as np

from morie.fn.kmstb import kamath_step_back_prompting


def test_kmstb_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_step_back_prompting(query, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmstb_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_step_back_prompting(query, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmstb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
