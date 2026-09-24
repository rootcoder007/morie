"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e4.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e4 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    b = rng.normal(0, 1, p)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert "method" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e4_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 10, 2
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    b = rng.normal(0, 1, p)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_4(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
