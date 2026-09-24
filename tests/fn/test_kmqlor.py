"""Tests for kmqlor.kamath_qlora_4bit."""

import pytest

from morie.fn import _array_core as np

from morie.fn.kmqlor import kamath_qlora_4bit


def test_kmqlor_basic():
    """Test basic functionality with a quantised base and a LoRA adapter."""
    rng = np.random.default_rng(42)
    out_dim, in_dim, r = 5, 4, 2
    codes = rng.integers(0, 16, (out_dim, in_dim))
    # blockwise absmax: one value per row
    absmax = rng.uniform(0.5, 2.0, out_dim)
    W0_nf4 = {"codes": codes, "absmax": absmax}
    A = rng.normal(0, 1, (r, in_dim))
    B = rng.normal(0, 1, (out_dim, r))
    alpha = 0.3
    x = rng.normal(0, 1, in_dim)
    result = kamath_qlora_4bit(W0_nf4, A, B, alpha, r, x)
    assert isinstance(result, dict)
    for key in ("h", "base", "delta", "W0_dequantized", "scaling", "rank",
                "n_trainable", "n_frozen_4bit", "estimate", "n", "method"):
        assert key in result
    h = result["h"]
    assert len(h) == out_dim
    assert all(np.isfinite(v) for v in h)
    W0d = result["W0_dequantized"]
    assert len(W0d) == out_dim
    assert all(len(row) == in_dim for row in W0d)
    assert result["n_frozen_4bit"] == out_dim * in_dim
    assert result["rank"] == r
    assert result["scaling"] == alpha / r
    assert isinstance(result["method"], str)


def test_kmqlor_edge():
    """Test edge cases: docstring example via tuple form and invalid input rejection."""
    codes = [[15, 0], [0, 15]]
    # blockwise absmax: one value per row
    absmax = [2.0, 2.0]
    W0_nf4 = (codes, absmax)
    A = [[1.0, 0.0]]
    B = [[0.0], [1.0]]
    result = kamath_qlora_4bit(W0_nf4, A, B, 1.0, 1, [1.0, 0.0])
    assert isinstance(result, dict)
    assert result["h"] == [2.0, -1.0]
    assert result["W0_dequantized"] == [[2.0, -2.0], [-2.0, 2.0]]
    assert result["rank"] == 1
    assert result["scaling"] == 1.0
    assert result["n_frozen_4bit"] == 4
    # Invalid: dict missing required 'absmax' key
    with pytest.raises(ValueError):
        kamath_qlora_4bit({"codes": [[15, 0]]}, A, B, 1.0, 1, [1.0, 0.0])
    # Invalid: dict missing required 'codes' key
    with pytest.raises(ValueError):
        kamath_qlora_4bit({"absmax": [1.0, 1.0]}, A, B, 1.0, 1, [1.0, 0.0])
