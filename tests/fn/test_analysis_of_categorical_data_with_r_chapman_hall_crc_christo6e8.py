"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e8.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_8."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e8 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_8,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e8_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # replicate estimates N_hat_i^(r) for r = 1, ..., R
    replicate_estimates = [float(x) for x in rng.normal(100.0, 5.0, 20)]
    # full estimate N_hat_i
    full_estimate = float(np.mean(replicate_estimates))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_8(
        replicate_estimates, full_estimate
    )
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert value >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e8_edge():
    """Test edge cases with a small number of replicates."""
    rng = np.random.default_rng(7)
    replicate_estimates = [float(x) for x in rng.normal(50.0, 1.0, 3)]
    full_estimate = float(np.mean(replicate_estimates))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_8(
        replicate_estimates, full_estimate
    )
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
