"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e1.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e1 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e1_basic():
    """Test basic functionality."""
    pi = 0.05
    se = 0.95
    sp = 0.99
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1(pi, se, sp)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e1_edge():
    """Test edge cases."""
    pi = 0.50
    se = 0.90
    sp = 0.95
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_1(pi, se, sp)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0
