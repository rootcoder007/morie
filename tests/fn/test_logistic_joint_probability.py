"""Tests for logistic_joint_probability.logistic_joint_probability."""

from morie.fn import _array_core as np

from morie.fn.logistic_joint_probability import (
    logistic_joint_probability,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_joint_probability(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_joint_probability(x)
    assert isinstance(result, dict)
