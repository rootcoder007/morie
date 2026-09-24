"""Tests for hmlcos.geron_cosine_annealing."""

from morie.fn import _array_core as np

from morie.fn.hmlcos import geron_cosine_annealing


def test_hmlcos_basic():
    """Test basic functionality."""
    t = 0.5
    T = 5
    eta_max = 0.5
    result = geron_cosine_annealing(t, T, eta_max)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmlcos_edge():
    """Test edge cases."""
    t = 0.5
    T = 5
    eta_max = 0.5
    result = geron_cosine_annealing(t, T, eta_max)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmlcos as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
