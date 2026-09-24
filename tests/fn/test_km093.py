"""Tests for km093.kamath_ch6_honest_score."""

from morie.fn import _array_core as np

from morie.fn.km093 import kamath_ch6_honest_score


def test_km093_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    words = ["good", "bad", "hate", "love", "ok", "fine", "evil", "nice"]
    Yhat = [[rng.choice(words) for _ in range(5)] for _ in range(4)]
    k = 5
    result = kamath_ch6_honest_score(Yhat, k, hurtlex={"bad", "hate", "evil"})
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n_hurtful" in result
    assert "n_completions" in result
    assert "per_prompt" in result
    assert result["k"] == 5
    assert result["n"] == 4
    assert result["n_completions"] == 20
    assert 0.0 <= result["estimate"] <= 1.0
    assert len(result["per_prompt"]) == 4
    assert all(0 <= c <= k for c in result["per_prompt"])


def test_km093_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    words = ["good", "bad", "hate", "love", "ok", "fine", "evil", "nice"]
    # Edge case: all hits via callable hurtlex
    Yhat = [[rng.choice(words) for _ in range(3)] for _ in range(2)]
    result = kamath_ch6_honest_score(
        Yhat, 3, hurtlex=lambda y: y in {"bad", "hate", "evil"})
    assert isinstance(result, dict)
    assert result["k"] == 3
    assert result["n"] == 2
    assert result["n_completions"] == 6
    assert 0.0 <= result["estimate"] <= 1.0
    # Edge case: invalid k raises
    with pytest.raises(ValueError):
        kamath_ch6_honest_score([["a", "b", "c"]], 0, hurtlex={"a"})
    # Edge case: empty Yhat raises
    with pytest.raises(ValueError):
        kamath_ch6_honest_score([], 3, hurtlex={"a"})


import pytest
