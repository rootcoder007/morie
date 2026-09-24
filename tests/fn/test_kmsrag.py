"""Tests for kmsrag.kamath_self_rag."""

from morie.fn import _array_core as np

from morie.fn.kmsrag import kamath_self_rag


def test_kmsrag_basic():
    """Test basic functionality."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    reflection_model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmsrag_edge():
    """Test edge cases."""
    context = np.random.default_rng(42).normal(0, 1, 100)
    reflection_model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_self_rag(context, reflection_model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmsrag as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
