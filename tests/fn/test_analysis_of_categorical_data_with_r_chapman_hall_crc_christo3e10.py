"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e10.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e10 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    bj0 = 0.5
    bjs = [0.1, 0.2, -0.3]
    xs = rng.normal(0, 1, 3)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10(bj0, bjs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e10_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    bj0 = -0.2
    bjs = [0.5, 0.5]
    xs = rng.normal(0, 1, 2)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_10(bj0, bjs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
