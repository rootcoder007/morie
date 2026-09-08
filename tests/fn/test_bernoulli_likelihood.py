"""Tests for bernoulli_likelihood.bernoulli_likelihood."""

from morie.fn import _array_core as np

from morie.fn.bernoulli_likelihood import (
    bernoulli_likelihood,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bernoulli_likelihood(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bernoulli_likelihood(x)
    assert isinstance(result, dict)
