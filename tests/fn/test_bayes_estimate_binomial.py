"""Tests for bayes_estimate_binomial.bayes_estimate_binomial."""

from morie.fn import _array_core as np

from morie.fn.bayes_estimate_binomial import (
    bayes_estimate_binomial,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_estimate_binomial(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_estimate_binomial(x)
    assert isinstance(result, dict)
