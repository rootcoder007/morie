"""Tests for hmfad.geron_forward_autodiff."""

import math

from morie.fn import _array_core as np
from morie.fn.hmfad import geron_forward_autodiff


def test_hmfad_basic():
    """Test basic functionality."""
    f = lambda v: v[0] ** 2
    result = geron_forward_autodiff(f, [3.0])
    assert isinstance(result, dict)
    assert "value" in result
    assert "grad" in result
    assert "n_passes" in result
    assert math.isclose(result["value"], 9.0)
    assert math.isclose(result["grad"][0], 6.0)
    assert result["n_passes"] == 1


def test_hmfad_edge():
    """Test edge cases."""
    f = lambda v: v[0] * v[1] + v[0].exp()
    result = geron_forward_autodiff(f, [0.0, 3.0])
    assert isinstance(result, dict)
    assert "value" in result
    assert "grad" in result
    assert "n_passes" in result
    assert math.isclose(result["value"], 1.0)
    assert math.isclose(result["grad"][0], 4.0)
    assert math.isclose(result["grad"][1], 0.0)
    assert result["n_passes"] == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmfad as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
