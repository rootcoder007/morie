"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e50.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e50 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e50_basic():
    """Test basic functionality."""
    b1 = 0.5
    var_b1 = 0.04
    c = 1.0
    z = 1.96
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50(b1, var_b1, c, z)
    assert isinstance(result, dict)
    assert "or" in result
    assert "method" in result
    assert "value" in result
    assert math.isfinite(float(result["or"]))


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e50_edge():
    """Test edge cases."""
    b1 = -0.3
    var_b1 = 0.01
    c = 2.0
    z = 1.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_50(b1, var_b1, c, z)
    assert isinstance(result, dict)
    assert "or" in result
    assert math.isfinite(float(result["or"]))
