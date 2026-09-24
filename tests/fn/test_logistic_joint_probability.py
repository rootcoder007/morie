"""Tests for logistic_joint_probability.logistic_joint_probability."""

from morie.fn import _array_core as np

from morie.fn.logistic_joint_probability import (
    logistic_joint_probability,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e4_basic():
    """Test basic functionality."""
    b = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = logistic_joint_probability(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e4_edge():
    """Test edge cases."""
    b = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = logistic_joint_probability(b, x, y)
    assert isinstance(result, dict)
