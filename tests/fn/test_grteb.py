"""Tests for grteb.geron_transformer_encoder_block."""

from morie.fn import _array_core as np

from morie.fn.grteb import geron_transformer_encoder_block


def test_grteb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    T, d_model, n_heads, d_ff = 6, 4, 2, 8
    d_k = d_model // n_heads

    x = rng.normal(0, 1, (T, d_model))

    WQ = [rng.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
    WK = [rng.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
    WV = [rng.normal(0, 0.1, (d_model, d_k)) for _ in range(n_heads)]
    WO = rng.normal(0, 0.1, (n_heads * d_k, d_model))
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    mha_weights = {
        "WQ": WQ, "WK": WK, "WV": WV, "WO": WO,
        "gamma": gamma1, "beta": beta1,
    }

    W1 = rng.normal(0, 0.1, (d_model, d_ff))
    W2 = rng.normal(0, 0.1, (d_ff, d_model))
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)
    ffn_weights = {
        "W1": W1, "W2": W2,
        "gamma": gamma2, "beta": beta2,
    }

    result = geron_transformer_encoder_block(x, mha_weights, ffn_weights)

    payload = result.payload if hasattr(result, "payload") else result
    assert "output" in payload
    assert "attention_output" in payload
    assert "attention_weights" in payload
    assert "hidden" in payload
    assert "ffn_output" in payload
    assert "estimate" in payload
    assert "n" in payload
    assert "method" in payload

    assert len(payload["output"]) == T
    assert len(payload["output"][0]) == d_model
    assert len(payload["attention_output"]) == T
    assert len(payload["hidden"]) == T
    assert len(payload["ffn_output"]) == T
    assert payload["n"] == T


def test_grteb_edge():
    """Test edge cases with minimal valid input."""
    rng = np.random.default_rng(0)
    T, d_model, n_heads, d_ff = 3, 4, 1, 4
    d_k = d_model // n_heads

    WQ = [rng.normal(0, 0.1, (d_model, d_k))]
    WK = [rng.normal(0, 0.1, (d_model, d_k))]
    WV = [rng.normal(0, 0.1, (d_model, d_k))]
    WO = rng.normal(0, 0.1, (n_heads * d_k, d_model))
    mha_weights = {"WQ": WQ, "WK": WK, "WV": WV, "WO": WO}

    W1 = np.zeros((d_model, d_ff))
    W2 = np.zeros((d_ff, d_model))
    ffn_weights = {"W1": W1, "W2": W2}

    x = rng.normal(0, 1, (T, d_model))
    result = geron_transformer_encoder_block(x, mha_weights, ffn_weights)

    payload = result.payload if hasattr(result, "payload") else result
    assert "output" in payload
    assert "attention_output" in payload
    assert "n" in payload
    assert len(payload["output"]) == T
    assert len(payload["output"][0]) == d_model
    assert payload["n"] == T


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grteb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
