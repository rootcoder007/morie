"""Tests for grptq.geron_static_ptq."""

from morie.fn import _array_core as np

from morie.fn.grptq import geron_static_ptq


def test_grptq_basic():
    """Test basic functionality."""
    model = [lambda a: 2 * a, lambda a: a + 1]
    calibration_data = [[1.0], [2.0]]
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grptq_edge():
    """Test edge cases."""
    model = [lambda a: 2 * a, lambda a: a + 1]
    calibration_data = [[1.0], [2.0]]
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grptq as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
