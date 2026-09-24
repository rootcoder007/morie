"""Tests for grffn.geron_transformer_feedforward."""

import math

from morie.fn import _array_core as np

from morie.fn.grffn import geron_transformer_feedforward


def test_grffn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d = 8
    d_ff = 16
    x = rng.normal(0, 1, d)
    W1 = rng.normal(0, 1, (d, d_ff))
    b1 = rng.normal(0, 1, d_ff)
    W2 = rng.normal(0, 1, (d_ff, d))
    b2 = rng.normal(0, 1, d)
    result = geron_transformer_feedforward(x, W1, b1, W2, b2)
    assert isinstance(result, dict)
    assert "output" in result
    assert "hidden" in result
    assert "d_model" in result
    assert "d_ff" in result
    assert "expansion_ratio" in result
    assert "sparsity" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert isinstance(result["output"], list)
    assert len(result["output"]) == d
    assert isinstance(result["hidden"], list)
    assert len(result["hidden"]) == d_ff
    assert 0.0 <= result["sparsity"] <= 1.0
    assert math.isfinite(result["expansion_ratio"])
    assert result["d_model"] == d
    assert result["d_ff"] == d_ff


def test_grffn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    d = 4
    d_ff = 8
    T = 3
    # Test with (T, d) shaped input -- the FFN is applied position-wise.
    x = rng.normal(0, 1, (T, d))
    W1 = rng.normal(0, 1, (d, d_ff))
    b1 = 0.0
    W2 = rng.normal(0, 1, (d_ff, d))
    b2 = 0.0
    result = geron_transformer_feedforward(x, W1, b1, W2, b2)
    assert isinstance(result, dict)
    assert "output" in result
    assert "hidden" in result
    assert "sparsity" in result
    assert "d_model" in result
    assert "d_ff" in result
    assert "expansion_ratio" in result
    assert isinstance(result["output"], list)
    assert len(result["output"]) == T
    assert isinstance(result["hidden"], list)
    assert len(result["hidden"]) == T
    assert 0.0 <= result["sparsity"] <= 1.0
    assert math.isfinite(result["expansion_ratio"])
    assert result["d_model"] == d
    assert result["d_ff"] == d_ff


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grffn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
