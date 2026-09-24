"""Tests for km139.kamath_ch9_simvlm_prefixlm."""

import math

from morie.fn import _array_core as np

from morie.fn.km139 import kamath_ch9_simvlm_prefixlm


def test_km139_basic():
    """Test basic functionality with a 1-D sequence of probabilities."""
    # Docstring example: x = [0.5, 0.5, 0.25], T_p = 1
    # Suffix = [0.5, 0.25]; -log(0.5) - log(0.25) = log(2) + log(4) = log(8)
    x = [0.5, 0.5, 0.25]
    T_p = 1
    result = kamath_ch9_simvlm_prefixlm(None, x, T_p)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert abs(result["estimate"] - math.log(8)) < 1e-12
    assert result["prefix_length"] == 1
    assert result["n_suffix_tokens"] == 2
    assert result["n"] == 1
    assert "per_sequence" in result
    assert len(result["per_sequence"]) == 1
    assert math.isfinite(result["per_sequence"][0])


def test_km139_edge():
    """Test with a callable theta and a 2-D batch input."""
    rng = np.random.default_rng(42)
    # Batch of n=4 sequences, each with p=6 token probabilities in (0, 1)
    x = rng.uniform(0.01, 0.99, (4, 6))
    # A callable that returns the input probabilities as-is
    def theta(inp):
        return inp
    T_p = 2
    result = kamath_ch9_simvlm_prefixlm(theta, x, T_p)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["prefix_length"] == 2
    assert result["n_suffix_tokens"] == 4
    assert result["n"] == 4
    assert "per_sequence" in result
    assert len(result["per_sequence"]) == 4
    for v in result["per_sequence"]:
        assert math.isfinite(v)
