"""Tests for km104.kamath_ch6_affect_lm."""
import math
import pytest

from morie.fn import _array_core as np
from morie.fn.km104 import kamath_ch6_affect_lm


def test_km104_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    vocab_size = 5
    hidden_c = 3
    hidden_e = 2

    U = rng.normal(0, 1, (vocab_size, hidden_c))
    V = rng.normal(0, 1, (vocab_size, hidden_e))
    c = rng.normal(0, 1, hidden_c)
    e = rng.normal(0, 1, hidden_e)
    b = rng.normal(0, 1, vocab_size)

    result = kamath_ch6_affect_lm(U, V, None, None, c, e, 0.7, b)

    # Returned object should be dict‑like and contain the documented keys
    assert isinstance(result, dict)
    assert "p" in result
    assert "affect_term" in result
    assert "argmax" in result
    assert "beta" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # Probability vector should have the right length and sum to 1
    p = result["p"]
    assert len(p) == vocab_size
    assert math.isclose(sum(p), 1.0, abs_tol=1e-6)

    # The estimated probability (max of p) must be finite
    assert math.isfinite(result["estimate"])

    # n should match the vocabulary size
    assert result["n"] == vocab_size


def test_km104_edge():
    """Test that a non‑finite beta raises ValueError as documented."""
    rng = np.random.default_rng(0)

    vocab_size = 4
    hidden_c = 2
    hidden_e = 2

    U = rng.normal(0, 1, (vocab_size, hidden_c))
    V = rng.normal(0, 1, (vocab_size, hidden_e))
    c = rng.normal(0, 1, hidden_c)
    e = rng.normal(0, 1, hidden_e)
    b = rng.normal(0, 1, vocab_size)

    with pytest.raises(ValueError):
        kamath_ch6_affect_lm(U, V, None, None, c, e, math.inf, b)
