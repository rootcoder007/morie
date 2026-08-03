"""Tests for pearson_chi2_two_groups.pearson_chi2_two_groups."""

from morie.fn import _array_core as np

from morie.fn.pearson_chi2_two_groups import (
    pearson_chi2_two_groups,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pearson_chi2_two_groups(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pearson_chi2_two_groups(x)
    assert isinstance(result, dict)
