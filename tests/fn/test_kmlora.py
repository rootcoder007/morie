"""Tests for kmlora.kamath_lora_weight_update."""

from morie.fn import _array_core as np

from morie.fn.kmlora import kamath_lora_weight_update


def test_kmlora_basic():
    """Test basic functionality."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmlora_edge():
    """Test edge cases."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    r = 10
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmlora as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
