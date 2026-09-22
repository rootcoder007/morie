"""Tests for ca2e7.ca_chapter_2_equation_7."""

from morie.fn import _array_core as np

from morie.fn.ca2e7 import ca_chapter_2_equation_7


def test_ca2e7_basic():
    """Test basic functionality with scalar correlations."""
    # Formula: b_x1 = ((r_y1 - r_y2*r_12)/(1 - r_12^2)) * (s_y/s_1)
    r_y1, r_y2, r_12 = 0.5, 0.3, 0.4
    s_y, s_1, s_2 = 10.0, 2.0, 3.0

    expected = ((r_y1 - r_y2 * r_12) / (1 - r_12**2)) * (s_y / s_1)

    result = ca_chapter_2_equation_7(r_y1, r_y2, r_12, s_y, s_1, s_2)

    assert isinstance(result, dict)
    assert "b1" in result
    assert result["b1"] == expected


def test_ca2e7_edge():
    """Test edge case with zero standard deviation ratio."""
    r_y1, r_y2, r_12 = 0.5, 0.3, 0.4
    s_y, s_1, s_2 = 0.0, 2.0, 3.0

    expected = ((r_y1 - r_y2 * r_12) / (1 - r_12**2)) * (s_y / s_1)

    result = ca_chapter_2_equation_7(r_y1, r_y2, r_12, s_y, s_1, s_2)

    assert isinstance(result, dict)
    assert result["b1"] == expected
