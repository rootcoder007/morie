"""Tests for grdyq.geron_dynamic_quantization."""

from morie.fn import _array_core as np

from morie.fn.grdyq import geron_dynamic_quantization


def test_grdyq_basic():
    """Test basic functionality."""
    x = [[0.5, -1.0], [0.25, 1.0]]
    w = [[2.0], [-2.0]]
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdyq_edge():
    """Test edge cases."""
    x = [[0.5, -1.0], [0.25, 1.0]]
    w = [[2.0], [-2.0]]
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grdyq as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
