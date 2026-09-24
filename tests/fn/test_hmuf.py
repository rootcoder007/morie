"""Tests for hmuf.geron_underfitting."""

from morie.fn import _array_core as np

from morie.fn.hmuf import geron_underfitting


def test_hmuf_basic():
    """Test basic functionality."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_underfitting(train_err, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmuf_edge():
    """Test edge cases."""
    train_err = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_underfitting(train_err, threshold)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmuf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
