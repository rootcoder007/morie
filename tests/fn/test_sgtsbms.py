"""Tests for sgtsbms.sgt_sbm_spectral_estimate."""

from morie.fn import _array_core as np

from morie.fn.sgtsbms import sgt_sbm_spectral_estimate


def test_sgtsbms_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_sbm_spectral_estimate(A, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_sgtsbms_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_sbm_spectral_estimate(A, K)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.sgtsbms as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
