"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e20.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e20 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e20_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.1
    bs = rng.normal(0, 1, 3)
    xs = rng.normal(0, 1, 3)
    random_intercept = 0.2

    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20(
        b0, bs, xs, random_intercept
    )

    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert "method" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e20_edge():
    """Test edge cases."""
    # All zeros should still produce a finite value
    b0 = 0.0
    bs = [0.0, 0.0, 0.0]
    xs = [0.0, 0.0, 0.0]
    random_intercept = 0.0

    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_20(
        b0, bs, xs, random_intercept
    )

    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert "method" in result
