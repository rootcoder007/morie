"""Tests for contingency_pmf.contingency_pmf."""

from morie.fn import _array_core as np

from morie.fn.contingency_pmf import (
    contingency_pmf,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = contingency_pmf(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = contingency_pmf(x)
    assert isinstance(result, dict)
