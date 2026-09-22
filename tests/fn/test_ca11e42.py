"""Tests for ca11e42.ca_chapter_11_equation_42."""

from morie.fn import _array_core as np

from morie.fn.ca11e42 import ca_chapter_11_equation_42


def test_ca11e42_basic():
    """Test basic functionality."""
    q = 50.0
    df = 10
    result = ca_chapter_11_equation_42(q, df)
    assert isinstance(result, dict)
    assert "value" in result
    # Independently compute the documented formula: I^2 = ((Q - df) / Q) * 100
    expected = ((q - df) / q) * 100
    assert result["value"] == expected


def test_ca11e42_edge():
    """Test edge cases."""
    q = 10.0
    df = 5
    result = ca_chapter_11_equation_42(q, df)
    assert isinstance(result, dict)
    assert "value" in result
    expected = ((q - df) / q) * 100
    assert result["value"] == expected
