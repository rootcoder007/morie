"""Tests for hmrad.geron_reverse_autodiff."""

import doctest as _doctest
import math

import morie.fn.hmrad as _doctest_module
from morie.fn import _array_core as np
from morie.fn.hmrad import geron_reverse_autodiff


def test_hmrad_basic():
    """Test basic functionality."""
    f = lambda v: v[0] * v[1]
    x = [3.0, 4.0]
    result = geron_reverse_autodiff(f, x)
    assert isinstance(result, dict)
    assert "gradient" in result
    assert "value" in result
    assert "n_passes" in result
    assert "n" in result
    assert "method" in result
    assert result["n_passes"] == 2
    assert math.isfinite(float(result["value"]))
    grad = list(result["gradient"])
    assert len(grad) == 2
    assert math.isfinite(float(grad[0]))
    assert math.isfinite(float(grad[1]))


def test_hmrad_edge():
    """Test edge cases."""
    f = lambda v: v[0].tanh() ** 2
    x = [0.0]
    result = geron_reverse_autodiff(f, x)
    assert isinstance(result, dict)
    assert "gradient" in result
    assert "value" in result
    assert "n" in result
    assert result["n"] == 1
    assert math.isfinite(float(result["value"]))
    grad = list(result["gradient"])
    assert len(grad) == 1
    assert math.isfinite(float(grad[0]))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
