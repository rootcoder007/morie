"""Tests for kmlb.kamath_moe_load_balance_loss."""

from morie.fn import _array_core as np

from morie.fn.kmlb import kamath_moe_load_balance_loss


def test_kmlb_basic():
    """Test basic functionality."""
    fractions = 1
    gate_means = 1
    N = 1
    alpha = 0.5
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmlb_edge():
    """Test edge cases."""
    fractions = 1
    gate_means = 1
    N = 1
    alpha = 0.5
    result = kamath_moe_load_balance_loss(fractions, gate_means, N, alpha)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmlb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
