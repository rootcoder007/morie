"""Tests for grhbb.geron_hebb_rule."""

from morie.fn import _array_core as np

from morie.fn.grhbb import geron_hebb_rule


def test_grhbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grhbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grhbb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
