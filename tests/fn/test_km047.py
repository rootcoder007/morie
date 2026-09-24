"""Tests for km047.kamath_ch3_translate_prefix_prompt."""

import pytest

from morie.fn.km047 import kamath_ch3_translate_prefix_prompt


def test_km047_basic():
    """Test basic functionality with a string and a slot."""
    x = "The cat sleeps."
    z = "Le chat dort."
    result = kamath_ch3_translate_prefix_prompt(x, z)
    assert isinstance(result, dict)
    assert "prompt" in result
    assert isinstance(result["prompt"], str)
    assert x in result["prompt"]
    assert "slot_filled" in result
    assert result["slot_filled"] is True


def test_km047_edge():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError):
        kamath_ch3_translate_prefix_prompt("")
