"""Tests for grmha.geron_multi_head_attention."""

import math

from morie.fn import _array_core as np

from morie.fn.grmha import geron_multi_head_attention


def test_grmha_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d_model = 4
    n_heads = 2
    Tq, Tk = 3, 5
    d_out = 4

    Q = rng.normal(0, 1, (Tq, d_model))
    K = rng.normal(0, 1, (Tk, d_model))
    V = rng.normal(0, 1, (Tk, d_model))
    WQ = rng.normal(0, 1, (d_model, d_model))
    WK = rng.normal(0, 1, (d_model, d_model))
    WV = rng.normal(0, 1, (d_model, d_model))
    WO = rng.normal(0, 1, (d_model, d_out))

    result = geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, n_heads)

    assert isinstance(result, dict)
    for key in (
        "output",
        "head_outputs",
        "attention_weights",
        "concat",
        "d_head",
        "n_heads",
        "estimate",
        "n",
        "method",
    ):
        assert key in result

    assert result["n_heads"] == n_heads
    assert result["d_head"] == d_model // n_heads

    # attention_weights: h matrices of shape (Tq, Tk) whose rows are distributions
    atts = result["attention_weights"]
    assert len(atts) == n_heads
    for A in atts:
        assert len(A) == Tq
        assert len(A[0]) == Tk
        for row in A:
            assert math.isclose(sum(row), 1.0, abs_tol=1e-6)

    # output has shape (Tq, d_out)
    assert len(result["output"]) == Tq
    assert len(result["output"][0]) == d_out


def test_grmha_edge():
    """Test edge case: a single head with identity projections on the smallest valid width."""
    I = [[1.0, 0.0], [0.0, 1.0]]
    Kv = [[1.0, 0.0], [0.0, 1.0]]
    # d_model = 2, h = 1, d_head = 2, d_out = 2
    result = geron_multi_head_attention([[1.0, 0.0]], Kv, Kv, I, I, I, I, h=1)

    assert isinstance(result, dict)
    for key in (
        "output",
        "head_outputs",
        "attention_weights",
        "concat",
        "d_head",
        "n_heads",
        "estimate",
        "n",
        "method",
    ):
        assert key in result

    assert result["n_heads"] == 1
    assert result["d_head"] == 2

    # exactly one attention matrix, rows are distributions
    atts = result["attention_weights"]
    assert len(atts) == 1
    assert len(atts[0]) == 1
    assert len(atts[0][0]) == 2
    assert math.isclose(sum(atts[0][0]), 1.0, abs_tol=1e-6)

    # output has shape (Tq=1, d_out=2) and all entries are finite
    assert len(result["output"]) == 1
    assert len(result["output"][0]) == 2
    for v in result["output"][0]:
        assert math.isfinite(v)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grmha as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
