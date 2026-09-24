"""Tests for kmglv.kamath_glove_cost."""

import math

from morie.fn import _array_core as np

from morie.fn.kmglv import kamath_glove_cost


def test_kmglv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    V, C, d = 40, 3, 4
    # Co-occurrence counts must be non-negative
    X = np.abs(rng.normal(0, 1, (V, C)))
    W = rng.normal(0, 1, (V, d))
    W_tilde = rng.normal(0, 1, (C, d))
    b = rng.normal(0, 1, V)
    b_tilde = rng.normal(0, 1, C)
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde,
                               x_max=100.0, alpha=0.75)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n_nonzero" in result
    assert "n" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0
    assert result["n_nonzero"] >= 1
    assert result["n"] == V * C


def test_kmglv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    V, C, d = 5, 4, 3
    X = np.abs(rng.normal(0, 1, (V, C)))
    W = rng.normal(0, 1, (V, d))
    W_tilde = rng.normal(0, 1, (C, d))
    b = rng.normal(0, 1, V)
    b_tilde = rng.normal(0, 1, C)
    result = kamath_glove_cost(X, W, W_tilde, b, b_tilde,
                               x_max=50.0, alpha=0.5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmglv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
