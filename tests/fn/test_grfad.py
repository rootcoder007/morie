"""Tests for grfad.geron_forward_mode_autodiff."""

from morie.fn import _array_core as np

from morie.fn.grfad import geron_forward_mode_autodiff


def test_grfad_basic():
    """Test basic functionality."""
    x = 1.5
    x_prime = 1.0
    f = lambda z: z ** 4
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfad_edge():
    """Test edge cases."""
    x = 1.5
    x_prime = 1.0
    f = lambda z: z ** 4
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grfad as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
