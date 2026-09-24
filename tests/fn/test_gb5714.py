"""Tests for gb5714.gibbons_wsrt_sampsize."""

from morie.fn import _array_core as np

from morie.fn.gb5714 import gibbons_wsrt_sampsize


def test_gb5714_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb5714_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gb5714 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
