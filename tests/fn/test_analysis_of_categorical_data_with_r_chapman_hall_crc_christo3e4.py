"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e4.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e4 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_4,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = 3
    bj0 = 0.5
    bjs = rng.normal(0, 1, p)
    xs = rng.normal(0, 1, p)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_4(bj0, bjs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    value = result["value"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e4_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = 2
    bj0 = -1.0
    bjs = [0.1, 0.2]
    xs = rng.normal(0, 1, p)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_4(bj0, bjs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    value = result["value"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value)
