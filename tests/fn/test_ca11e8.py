"""Tests for ca11e8.ca_chapter_11_equation_8."""

from morie.fn import _array_core as np

from morie.fn.ca11e8 import ca_chapter_11_equation_8


def test_ca11e8_basic():
    """Test basic functionality."""
    a, b, c, d = 50, 30, 20, 60
    result = ca_chapter_11_equation_8(a, b, c, d)
    assert isinstance(result, dict)
    assert "value" in result
    expected = (a / (a + b)) / (c / (c + d))
    assert result["value"] == expected


def test_ca11e8_edge():
    """Test edge cases."""
    # When a/(a+b) equals c/(c+d), the risk ratio is 1.
    a, b, c, d = 10, 10, 10, 10
    result = ca_chapter_11_equation_8(a, b, c, d)
    assert isinstance(result, dict)
    assert result["value"] == 1.0
    assert result["method"] == "Weisburd et al. (2022) eq. (11.8)"
