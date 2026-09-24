"""Tests for mle_variance_pi.mle_variance_pi."""

from morie.fn import _array_core as np

from morie.fn.mle_variance_pi import (
    mle_variance_pi,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e3_basic():
    """Test basic functionality."""
    pi_hat = 0.5
    n = 0.5
    result = mle_variance_pi(pi_hat, n)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e3_edge():
    """Test edge cases."""
    pi_hat = 0.5
    n = 0.5
    result = mle_variance_pi(pi_hat, n)
    assert isinstance(result, dict)
