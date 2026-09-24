"""Tests for km048.kamath_ch3_cloze_prompt_template."""

from morie.fn import _array_core as np

from morie.fn.km048 import kamath_ch3_cloze_prompt_template


def test_km048_basic():
    """Test basic functionality."""
    result = kamath_ch3_cloze_prompt_template("Loved it.", "great")
    assert isinstance(result, dict)
    assert "prompt" in result
    assert isinstance(result["prompt"], str)
    assert "Loved it." in result["prompt"]
    assert "great" in result["prompt"]


def test_km048_edge():
    """Test edge cases."""
    result = kamath_ch3_cloze_prompt_template("Cannot watch this.")
    assert isinstance(result, dict)
    assert "prompt" in result
    assert isinstance(result["prompt"], str)
    assert "Cannot watch this." in result["prompt"]
