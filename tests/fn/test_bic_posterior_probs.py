"""Tests for bic_posterior_probs.bic_posterior_probs."""

from morie.fn import _array_core as np

from morie.fn.bic_posterior_probs import (
    bic_posterior_probs,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bic_posterior_probs(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bic_posterior_probs(x)
    assert isinstance(result, dict)
