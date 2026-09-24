"""Tests for grrad.geron_reverse_mode_autodiff."""

from morie.fn import _array_core as np

from morie.fn.grrad import geron_reverse_mode_autodiff


def test_grrad_basic():
    """Test basic functionality."""
    # L = a * b with a=3, b=4: partials dL/da = b = 4, dL/db = a = 3
    graph = {"L": {"a": 4.0, "b": 3.0}}
    result = geron_reverse_mode_autodiff(graph, loss_grad=1.0)
    assert isinstance(result, dict)
    assert "gradients" in result
    assert result["gradients"]["a"] == 4.0
    assert result["gradients"]["b"] == 3.0


def test_grrad_edge():
    """Test edge cases."""
    # A node feeding two paths accumulates both contributions:
    # L = u + v, u = 2x, v = 3x  =>  dL/dx = 2 + 3 = 5
    graph = {"L": {"u": 1.0, "v": 1.0}, "u": {"x": 2.0}, "v": {"x": 3.0}}
    result = geron_reverse_mode_autodiff(graph)
    assert isinstance(result, dict)
    assert "gradients" in result
    assert result["gradients"]["x"] == 5.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grrad as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
