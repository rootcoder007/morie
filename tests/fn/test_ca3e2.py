"""Tests for ca3e2.ca_chapter_3_equation_2."""

from morie.fn import _array_core as np

from morie.fn.ca3e2 import ca_chapter_3_equation_2


def test_ca3e2_basic():
    """Test basic functionality with a single scalar R^2 value."""
    r2_x = 0.25
    result = ca_chapter_3_equation_2(r2_x)
    assert isinstance(result, dict)
    assert "value" in result
    # VIF = 1/(1 - R^2) computed independently from the documented formula.
    expected = 1.0 / (1.0 - r2_x)
    assert result["value"] == expected


def test_ca3e2_edge():
    """Test edge cases on the documented [0, 1) domain for R^2."""
    # R^2 = 0 -> VIF = 1 (no multicollinearity).
    r2_zero = 0.0
    result_zero = ca_chapter_3_equation_2(r2_zero)
    assert isinstance(result_zero, dict)
    assert "value" in result_zero
    assert result_zero["value"] == 1.0 / (1.0 - r2_zero)

    # R^2 approaching 1 -> VIF grows large, but stays finite for valid inputs.
    r2_high = 0.9
    result_high = ca_chapter_3_equation_2(r2_high)
    assert isinstance(result_high, dict)
    assert "value" in result_high
    assert result_high["value"] == 1.0 / (1.0 - r2_high)
