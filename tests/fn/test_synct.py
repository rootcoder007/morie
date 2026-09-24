"""Tests for synct.synthetic_control."""

from morie.fn import _array_core as np

from morie.fn.synct import synthetic_control


def test_synct_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated_unit = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_synct_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    unit_id = np.random.default_rng(42).normal(0, 1, 100)
    time_id = np.random.default_rng(42).normal(0, 1, 100)
    treated_unit = np.random.default_rng(42).normal(0, 1, 100)
    treatment_time = np.random.default_rng(42).normal(0, 1, 100)
    result = synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.synct as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
