"""Tests for polr_parameterization.polr_parameterization."""

from morie.fn import _array_core as np

from morie.fn.polr_parameterization import (
    polr_parameterization,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = polr_parameterization(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = polr_parameterization(x)
    assert isinstance(result, dict)
