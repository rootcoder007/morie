"""Tests for km078.kamath_ch6_alignment_function."""

import math

from morie.fn import _array_core as np

from morie.fn.km078 import kamath_ch6_alignment_function


def test_km078_basic():
    """Test basic functionality with 3way classification head."""
    a = "the cat sat"
    b = "a cat sat"
    f = lambda a, b: "ALIGNED"
    result = kamath_ch6_alignment_function(a, b, "3way", f=f)
    assert isinstance(result, dict)
    assert result["label"] == "ALIGNED"
    assert result["estimate"] == 1.0
    assert result["space"] == "3way"
    assert "labels" in result
    assert result["n"] == 2


def test_km078_edge():
    """Test edge cases with regression head."""
    f = lambda a, b: 0.25
    result = kamath_ch6_alignment_function("x", "y", "reg", f=f)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["space"] == "reg"
    assert result["label"] is None
    assert result["labels"] is None
