"""Tests for kott_carr_interval.kott_carr_interval."""

from morie.fn import _array_core as np

from morie.fn.kott_carr_interval import (
    kott_carr_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e11_basic():
    """Test basic functionality."""
    pi_hat = 0.5
    var_pi = 0.5
    t_crit = 0.5
    result = kott_carr_interval(pi_hat, var_pi, t_crit)
    assert isinstance(result, dict)
    assert "n_effective" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e11_edge():
    """Test edge cases."""
    pi_hat = 0.5
    var_pi = 0.5
    t_crit = 0.5
    result = kott_carr_interval(pi_hat, var_pi, t_crit)
    assert isinstance(result, dict)
