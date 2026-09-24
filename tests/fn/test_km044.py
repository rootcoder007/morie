"""Tests for km044.kamath_ch3_prompt_search_argmax."""

from morie.fn import _array_core as np

from morie.fn.km044 import kamath_ch3_prompt_search_argmax


def test_km044_basic():
    """Test basic functionality."""
    score = lambda s: float(len(s))
    result = kamath_ch3_prompt_search_argmax(
        "Paris is [z].", ["great", "terrible"], score)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "z_hat" in result
    assert "filled_prompt" in result
    assert "scores" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 2
    assert result["z_hat"] == "terrible"
    assert result["estimate"] == 18.0
    assert isinstance(result["scores"], dict)
    assert set(result["scores"].keys()) == {"great", "terrible"}


def test_km044_edge():
    """Test edge case: template without [z] slot uses append fallback."""
    score = lambda s: float(len(s))
    result = kamath_ch3_prompt_search_argmax(
        "hello", ["a", "bb", "ccc"], score)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "z_hat" in result
    assert "filled_prompt" in result
    assert "scores" in result
    assert "n" in result
    assert result["n"] == 3
    assert result["z_hat"] in ["a", "bb", "ccc"]
    assert isinstance(result["scores"], dict)
    assert set(result["scores"].keys()) == {"a", "bb", "ccc"}
    assert result["z_hat"] == "ccc"
