"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e3.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3."""

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e3 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e3_basic():
    """Test basic functionality."""
    pi = 0.10
    se = 0.95
    sp = 0.90
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3(pi, se, sp)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e3_edge():
    """Test edge cases."""
    pi = 0.05
    se = 0.99
    sp = 0.98
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_3(pi, se, sp)
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0
