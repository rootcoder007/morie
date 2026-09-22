"""Tests for ca2e10.ca_chapter_2_equation_10."""

from morie.fn import _array_core as np

from morie.fn.ca2e10 import ca_chapter_2_equation_10


def test_ca2e10_basic():
    """Test basic functionality with scalar arguments."""
    b = 0.5
    se = 0.1
    t_cv = 1.96
    result = ca_chapter_2_equation_10(b, se, t_cv)
    assert isinstance(result, dict)
    assert "lower" in result
    assert result["lower"] == b - se * t_cv
    expected_upper = b + se * t_cv
    assert result["upper"] == expected_upper
    assert "value" in result
    assert result["value"] == b - se * t_cv
    assert "method" in result


def test_ca2e10_edge():
    """Test edge case with zero-width interval."""
    b = -0.25
    se = 0.05
    t_cv = 0.0
    result = ca_chapter_2_equation_10(b, se, t_cv)
    assert isinstance(result, dict)
    assert result["lower"] == b - se * t_cv
    assert result["upper"] == b + se * t_cv
    assert result["lower"] == result["upper"] == b
