"""Tests for survey_proportion_variance.survey_proportion_variance."""

from morie.fn import _array_core as np

from morie.fn.survey_proportion_variance import (
    survey_proportion_variance,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_proportion_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_proportion_variance(x)
    assert isinstance(result, dict)
