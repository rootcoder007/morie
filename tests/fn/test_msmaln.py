"""Tests for msmaln.aalen_johansen."""

import math

from morie.fn import _array_core as np

from morie.fn.msmaln import aalen_johansen


def test_msmaln_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    time = rng.uniform(0, 10, n)
    cause = rng.integers(0, 3, n)  # 0 = censored, 1 or 2 = cause of event
    result = aalen_johansen(time, cause, n_causes=2)
    assert isinstance(result, dict)
    expected_keys = {"times", "cif", "overall_survival", "naive_km",
                     "overstatement", "partition_residual", "at_risk"}
    for key in expected_keys:
        assert key in result
    T = len(result["times"])
    assert len(result["cif"]) == 2
    assert len(result["cif"][0]) == T
    surv = result["overall_survival"]
    assert all(0 <= s <= 1 for s in surv)


def test_msmaln_edge():
    """Test edge case: single cause (no competing risks)."""
    rng = np.random.default_rng(123)
    n = 40
    time = rng.uniform(0, 10, n)
    cause = rng.integers(0, 2, n)  # 0 = censored, 1 = cause
    result = aalen_johansen(time, cause, n_causes=1)
    assert isinstance(result, dict)
    assert "times" in result
    assert "cif" in result
    assert len(result["cif"]) == 1
    # Partition: sum_k cif_k(t) + S(t) = 1 at every event time
    resid = result["partition_residual"]
    assert math.isfinite(resid)
    assert abs(resid) < 1e-9


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.msmaln as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
