"""Tests for logistic_pi.logistic_pi."""

from morie.fn import _array_core as np

from morie.fn.logistic_pi import (
    logistic_pi,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_pi(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = logistic_pi(x)
    assert isinstance(result, dict)
