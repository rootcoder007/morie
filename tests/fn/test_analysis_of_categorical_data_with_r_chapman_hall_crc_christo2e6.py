"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e6.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e6 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e6_basic():
    """Test basic functionality."""
    loglik_null = -50.0
    loglik_full = -40.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6(
        loglik_null, loglik_full
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    # loglik_full >= loglik_null, so -2*(LL0 - LLa) >= 0
    assert result["value"] >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e6_edge():
    """Test edge cases."""
    loglik_null = -10.0
    loglik_full = -10.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_6(
        loglik_null, loglik_full
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
