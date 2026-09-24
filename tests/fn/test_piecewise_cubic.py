"""Tests for piecewise_cubic.piecewise_cubic."""

import math

from morie.fn.piecewise_cubic import (
    piecewise_cubic,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e34_basic():
    """Test basic functionality."""
    knot = 0.0
    coef_left = [1.0, 2.0, 3.0, 4.0]
    coef_right = [0.5, 1.5, 2.5, 3.5]
    # x > knot: right coefficients are used
    result = piecewise_cubic(1.0, knot, coef_left, coef_right)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    # x < knot: left coefficients are used
    result_left = piecewise_cubic(-1.0, knot, coef_left, coef_right)
    assert isinstance(result_left, dict)
    assert "value" in result_left
    assert math.isfinite(result_left["value"])
    # x == knot: left branch selected by <=
    result_at = piecewise_cubic(0.0, knot, coef_left, coef_right)
    assert isinstance(result_at, dict)
    assert "value" in result_at
    assert math.isfinite(result_at["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e34_edge():
    """Test edge cases."""
    knot = 1.0
    coef_left = [0.0, 0.0, 0.0, 0.0]
    coef_right = [0.0, 0.0, 0.0, 0.0]
    # All zero coefficients -> cubic evaluates to 0 on either side
    result = piecewise_cubic(2.0, knot, coef_left, coef_right)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
    result2 = piecewise_cubic(-0.5, knot, coef_left, coef_right)
    assert isinstance(result2, dict)
    assert "value" in result2
    assert result2["value"] == 0.0
    result3 = piecewise_cubic(1.0, knot, coef_left, coef_right)
    assert isinstance(result3, dict)
    assert "value" in result3
    assert result3["value"] == 0.0
