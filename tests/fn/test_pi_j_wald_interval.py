"""Tests for pi_j_wald_interval.pi_j_wald_interval."""

from morie.fn import _array_core as np

from morie.fn.pi_j_wald_interval import (
    pi_j_wald_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e8_basic():
    """Test basic functionality."""
    pi_hat = 0.5
    var_pi = 0.5
    z = 0.5
    result = pi_j_wald_interval(pi_hat, var_pi, z)
    assert isinstance(result, dict)
    assert "lower" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e8_edge():
    """Test edge cases."""
    pi_hat = 0.5
    var_pi = 0.5
    z = 0.5
    result = pi_j_wald_interval(pi_hat, var_pi, z)
    assert isinstance(result, dict)
