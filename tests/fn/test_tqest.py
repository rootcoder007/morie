"""Tests for tqest.turboquant_estimate_scores."""

from morie.fn import _array_core as np

from morie.fn.tqest import turboquant_estimate_scores


def test_tqest_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_tildes = np.random.default_rng(42).normal(0, 1, 100)
    norms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_estimate_scores(q, k_tildes, norms, S)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tqest_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k_tildes = np.random.default_rng(42).normal(0, 1, 100)
    norms = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_estimate_scores(q, k_tildes, norms, S)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.tqest as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
