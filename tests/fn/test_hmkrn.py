"""Tests for hmkrn.geron_filter_kernel."""

from morie.fn import _array_core as np

from morie.fn.hmkrn import geron_filter_kernel


def test_hmkrn_basic():
    """Test basic functionality."""
    kh = np.random.default_rng(42).normal(0, 1, 100)
    kw = np.random.default_rng(42).normal(0, 1, 100)
    c_in = np.random.default_rng(42).normal(0, 1, 100)
    c_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_filter_kernel(kh, kw, c_in, c_out, seed)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmkrn_edge():
    """Test edge cases."""
    kh = np.random.default_rng(42).normal(0, 1, 100)
    kw = np.random.default_rng(42).normal(0, 1, 100)
    c_in = np.random.default_rng(42).normal(0, 1, 100)
    c_out = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_filter_kernel(kh, kw, c_in, c_out, seed)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmkrn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
