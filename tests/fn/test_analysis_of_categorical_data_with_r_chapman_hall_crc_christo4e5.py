"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e5.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e5 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e5_basic():
    """Test basic functionality with small valid inputs."""
    rng = np.random.default_rng(42)
    b0 = 0.5
    beta_x_i = float(rng.normal(0, 1))
    beta_z_j = float(rng.normal(0, 1))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5(
        b0, beta_x_i, beta_z_j
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    value = result["value"]
    assert math.isfinite(float(value))
    assert float(value) > 0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e5_edge():
    """Test edge case with minimal valid input sizes."""
    rng = np.random.default_rng(42)
    b0 = 0.0
    beta_x_i = float(rng.normal(0, 1))
    beta_z_j = float(rng.normal(0, 1))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_4_equation_5(
        b0, beta_x_i, beta_z_j
    )
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(float(value))
    assert float(value) > 0
