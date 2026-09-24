"""Tests for hwmul.holt_winters_mult."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.hwmul import holt_winters_mult


def test_hwmul_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    y = (10.0 + rng.normal(0.0, 1.0, 100)).tolist()
    result = holt_winters_mult(y, period=12, alpha=0.4, beta=0.1, gamma=0.3)
    assert isinstance(result, dict)
    for key in ("forecast", "level", "trend", "seasonal", "fitted", "residuals", "sse"):
        assert key in result
    assert math.isfinite(float(result["sse"]))


def test_hwmul_edge():
    """Test edge cases: non-positive data is rejected."""
    y = [1.0] * 23 + [0.0]
    with pytest.raises(ValueError):
        holt_winters_mult(y, period=12)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hwmul as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
