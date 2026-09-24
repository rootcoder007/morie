"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e4.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e4 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.5
    beta_x_i = float(rng.normal(0, 1))
    beta_z_j = float(rng.normal(0, 1))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4(
        b0, beta_x_i, beta_z_j
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e4_edge():
    """Test edge cases with zero coefficients."""
    b0 = 0.0
    beta_x_i = 0.0
    beta_z_j = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_4(
        b0, beta_x_i, beta_z_j
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
