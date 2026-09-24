"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e15.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_15."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e15 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_15,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e15_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = rng.normal(0, 1)
    beta_w_a = rng.normal(0, 0.5)
    beta_y_b = rng.normal(0, 0.5)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_15(
        b0, beta_w_a, beta_y_b
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["method"] == "Bilder & Loughin (2025) eq. (6.15)"


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e15_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    # Use zero values, which are valid inputs.
    b0 = 0.0
    beta_w_a = 0.0
    beta_y_b = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_15(
        b0, beta_w_a, beta_y_b
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["method"] == "Bilder & Loughin (2025) eq. (6.15)"
