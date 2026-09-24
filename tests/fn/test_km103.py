"""Tests for km103.kamath_ch6_lstm_softmax_word."""

import math

from morie.fn import _array_core as np

from morie.fn.km103 import kamath_ch6_lstm_softmax_word


def test_km103_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    V = 10
    d = 4
    U = rng.normal(0, 1, (V, d))
    f = None
    c_t_1 = rng.normal(0, 1, d)
    b = rng.normal(0, 1, V)
    result = kamath_ch6_lstm_softmax_word(U, f, c_t_1, b)
    assert isinstance(result, dict)
    assert "p" in result
    assert "logits" in result
    assert "argmax" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["p"]) == V
    assert len(result["logits"]) == V
    assert result["n"] == V
    assert math.isclose(sum(result["p"]), 1.0, abs_tol=1e-9)
    for p_val in result["p"]:
        assert 0.0 <= p_val <= 1.0
        assert math.isfinite(p_val)
    assert math.isclose(result["estimate"], max(result["p"]), abs_tol=1e-9)
    assert math.isfinite(result["estimate"])
    assert 0 <= result["argmax"] < V


def test_km103_edge():
    """Test edge cases: bias invariance."""
    rng = np.random.default_rng(42)
    V = 5
    d = 3
    U = rng.normal(0, 1, (V, d))
    f = None
    c_t_1 = rng.normal(0, 1, d)
    b = rng.normal(0, 1, V)
    result = kamath_ch6_lstm_softmax_word(U, f, c_t_1, b)
    shift = 2.5
    b_shifted = [bi + shift for bi in b]
    result_shifted = kamath_ch6_lstm_softmax_word(U, f, c_t_1, b_shifted)
    assert len(result["p"]) == V
    assert len(result_shifted["p"]) == V
    assert math.isclose(sum(result["p"]), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(result_shifted["p"]), 1.0, abs_tol=1e-9)
    for i in range(V):
        assert math.isclose(result["p"][i], result_shifted["p"][i], abs_tol=1e-9)
    assert result["argmax"] == result_shifted["argmax"]
