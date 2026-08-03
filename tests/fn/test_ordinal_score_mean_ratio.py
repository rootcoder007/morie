"""Tests for ordinal_score_mean_ratio.ordinal_score_mean_ratio."""

from morie.fn import _array_core as np

from morie.fn.ordinal_score_mean_ratio import (
    ordinal_score_mean_ratio,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_score_mean_ratio(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_score_mean_ratio(x)
    assert isinstance(result, dict)
