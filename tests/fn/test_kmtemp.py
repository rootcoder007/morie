"""Tests for kmtemp.kamath_temperature_sampling."""

from morie.fn import _array_core as np

from morie.fn.kmtemp import kamath_temperature_sampling


def test_kmtemp_basic():
    """Test basic functionality."""
    logits = 0.5
    T = 0.5
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "probabilities" in result


def test_kmtemp_edge():
    """Test edge cases."""
    logits = 0.5
    T = 0.5
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtemp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
