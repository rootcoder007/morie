"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e18.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e18 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e18_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.5
    b1 = 1.2
    x = float(rng.normal(0, 1))
    random_intercept = float(rng.normal(0, 1))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18(
        b0, b1, x, random_intercept
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e18_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    b0 = 0.0
    b1 = 1.0
    x = float(rng.normal(0, 1))
    random_intercept = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_18(
        b0, b1, x, random_intercept
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
