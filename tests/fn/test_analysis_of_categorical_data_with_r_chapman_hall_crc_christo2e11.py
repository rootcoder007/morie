"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e11.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e11 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e11_basic():
    """Test basic functionality."""
    b1 = 0.5
    var_b1 = 0.1
    c = 1.0
    z = 1.96
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11(b1, var_b1, c, z)
    assert isinstance(result, dict)
    assert "or" in result
    or_value = result["or"]
    assert math.isfinite(or_value)
    assert or_value > 0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e11_edge():
    """Test edge cases."""
    b1 = 0.0
    var_b1 = 0.05
    c = 2.0
    z = 2.576
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_11(b1, var_b1, c, z)
    assert isinstance(result, dict)
    assert "or" in result
    or_value = result["or"]
    assert math.isfinite(or_value)
    assert or_value > 0
