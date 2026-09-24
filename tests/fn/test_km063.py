"""Tests for km063.kamath_ch4_vera_forward."""

from morie.fn import _array_core as np
import math
from morie.fn.km063 import kamath_ch4_vera_forward


def test_km063_basic():
    """Test basic functionality."""
    d, k, r = 4, 3, 2
    rng = np.random.default_rng(42)
    W_0 = rng.normal(0, 1, (d, k))
    Lambda_b = rng.normal(0, 1, d)
    Lambda_d = rng.normal(0, 1, r)
    A = rng.normal(0, 1, (r, k))
    B = rng.normal(0, 1, (d, r))
    x = rng.normal(0, 1, k)
    result = kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x)
    assert isinstance(result, dict)
    assert "h" in result
    assert isinstance(result["h"], list)
    assert len(result["h"]) == d
    assert math.isfinite(result["estimate"])
    assert result["n_trainable"] == d + r
    assert result["n_trainable_lora"] == r * (d + k)
    assert result["r"] == r
    assert result["n"] == k


def test_km063_edge():
    """Test edge cases."""
    W_0 = [[1.0, 0.0], [0.0, 1.0]]
    Lambda_b = [3.0, 1.0]
    Lambda_d = [2.0]
    A = [[1.0, 0.0]]
    B = [[1.0], [0.0]]
    x = [1.0, 2.0]
    result = kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x)
    assert isinstance(result, dict)
    assert "h" in result
    assert isinstance(result["h"], list)
    assert len(result["h"]) == 2
    assert result["n_trainable"] == 3
    assert result["n_trainable_lora"] == 4
    assert math.isfinite(result["estimate"])
