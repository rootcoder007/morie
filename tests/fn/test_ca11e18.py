"""Tests for ca11e18.ca_chapter_11_equation_18."""

from morie.fn import _array_core as np

from morie.fn.ca11e18 import ca_chapter_11_equation_18


def test_ca11e18_basic():
    """Test basic functionality with a scalar ln(OR)."""
    x = 0.825
    result = ca_chapter_11_equation_18(x)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 0.825 / 1.65
    assert abs(result["value"] - expected) < 1e-12
    assert "method" in result


def test_ca11e18_edge():
    """Test edge case with ln(OR) = 0."""
    x = 0.0
    result = ca_chapter_11_equation_18(x)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
