"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e10.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10."""

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e10 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # 50 replicate survey proportion estimates (values in [0, 1])
    replicate_estimates = list(rng.uniform(0.2, 0.4, 50))
    # Full-sample estimate (a scalar proportion)
    full_estimate = float(np.mean(replicate_estimates))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10(
        replicate_estimates, full_estimate
    )
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    # Jackknife variance must be non-negative
    assert result["value"] >= 0.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e10_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Small number of replicates (R = 2)
    replicate_estimates = list(rng.uniform(0.1, 0.3, 2))
    full_estimate = 0.2
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_10(
        replicate_estimates, full_estimate
    )
    assert isinstance(result, dict)
    assert "value" in result
    import math
    assert math.isfinite(result["value"])
    assert result["value"] >= 0.0
