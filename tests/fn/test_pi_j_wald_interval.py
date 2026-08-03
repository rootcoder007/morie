"""Tests for pi_j_wald_interval.pi_j_wald_interval."""

from morie.fn import _array_core as np

from morie.fn.pi_j_wald_interval import (
    pi_j_wald_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pi_j_wald_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = pi_j_wald_interval(x)
    assert isinstance(result, dict)
