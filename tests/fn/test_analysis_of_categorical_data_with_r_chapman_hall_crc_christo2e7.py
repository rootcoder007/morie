"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e7.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e7 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e7_basic():
    """Test basic functionality."""
    loglik_null = -100.0
    loglik_full = -80.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7(
        loglik_null, loglik_full
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e7_edge():
    """Test edge cases."""
    loglik_null = -50.0
    loglik_full = -50.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_7(
        loglik_null, loglik_full
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
