"""Tests for kmtopk.kamath_moe_top_k_gating."""

from morie.fn import _array_core as np

from morie.fn.kmtopk import kamath_moe_top_k_gating


def test_kmtopk_basic():
    """Test basic functionality."""
    gates = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmtopk_edge():
    """Test edge cases."""
    gates = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_moe_top_k_gating(gates, k)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtopk as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
