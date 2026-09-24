"""Tests for llama.llama."""

import math

from morie.fn import _array_core as np

from morie.fn.llama import llama


def test_llama_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    d_in = 8
    d_attn = 8
    d_ff = 16
    tokens = rng.normal(0, 1, (n, d_in))
    model = {
        "Wq": rng.normal(0, 1, (d_in, d_attn)),
        "Wk": rng.normal(0, 1, (d_in, d_attn)),
        "Wv": rng.normal(0, 1, (d_in, d_attn)),
        "Wo": rng.normal(0, 1, (d_attn, d_in)),
        "W1": rng.normal(0, 1, (d_in, d_ff)),
        "W3": rng.normal(0, 1, (d_in, d_ff)),
        "W2": rng.normal(0, 1, (d_ff, d_in)),
    }
    g1 = rng.normal(0, 1, d_in)
    g2 = rng.normal(0, 1, d_in)
    result = llama(tokens, model, g1=g1, g2=g2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "out" in result
    assert "attn" in result
    assert "h" in result
    assert isinstance(result["out"], list)
    assert len(result["out"]) == n
    assert isinstance(result["h"], list)
    assert len(result["h"]) == n
    assert isinstance(result["attn"], list)
    assert len(result["attn"]) == n
    assert math.isfinite(result["estimate"])


def test_llama_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 1
    d_in = 4
    d_attn = 4
    d_ff = 8
    tokens = rng.normal(0, 1, (n, d_in))
    model = {
        "Wq": rng.normal(0, 1, (d_in, d_attn)),
        "Wk": rng.normal(0, 1, (d_in, d_attn)),
        "Wv": rng.normal(0, 1, (d_in, d_attn)),
        "Wo": rng.normal(0, 1, (d_attn, d_in)),
        "W1": rng.normal(0, 1, (d_in, d_ff)),
        "W3": rng.normal(0, 1, (d_in, d_ff)),
        "W2": rng.normal(0, 1, (d_ff, d_in)),
    }
    result = llama(tokens, model)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
