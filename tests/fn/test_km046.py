"""Tests for km046.kamath_ch3_prefix_prompt_template."""

from morie.fn import _array_core as np

from morie.fn.km046 import kamath_ch3_prefix_prompt_template


def test_km046_basic():
    """Test basic functionality."""
    x = "Cannot watch this movie."
    z = None
    result = kamath_ch3_prefix_prompt_template(x, z)
    assert isinstance(result, dict)
    assert "prompt" in result
    assert "slot_filled" in result
    assert result["prompt"] == "Cannot watch this movie. This movie is [z]"
    assert result["slot_filled"] is False


def test_km046_edge():
    """Test edge cases."""
    x = "Loved it."
    z = "great"
    result = kamath_ch3_prefix_prompt_template(x, z)
    assert isinstance(result, dict)
    assert "prompt" in result
    assert "slot_filled" in result
    assert result["prompt"] == "Loved it. This movie is great"
    assert result["slot_filled"] is True
