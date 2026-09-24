"""Tests for km113.kamath_ch8_perplexity."""

import math

from morie.fn import _array_core as np

from morie.fn.km113 import kamath_ch8_perplexity


def test_km113_basic():
    """Test basic functionality with array of token probabilities."""
    rng = np.random.default_rng(42)
    n_tokens = 100
    tokens = list(range(n_tokens))
    p_theta = rng.uniform(0.01, 1.0, n_tokens)
    result = kamath_ch8_perplexity(tokens, N=n_tokens, p_theta=p_theta)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0
    assert result["n"] == n_tokens
    assert math.isfinite(result["mean_nll"])
    assert result["mean_nll"] >= 0
    assert len(result["log_probs"]) == n_tokens


def test_km113_edge():
    """Test edge case from docstring example."""
    result = kamath_ch8_perplexity(["a", "b"], p_theta=[0.5, 0.5])
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 2.0
    assert result["n"] == 2
