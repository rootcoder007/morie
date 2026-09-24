"""Tests for pi_wald_interval.pi_wald_interval."""

from morie.fn import _array_core as np

from morie.fn.pi_wald_interval import (
    pi_wald_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e15_basic():
    """Test basic functionality."""
    xb = 0.5
    var_xb = 0.5
    z = 0.5
    result = pi_wald_interval(xb, var_xb, z)
    assert isinstance(result, dict)
    assert "pi" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e15_edge():
    """Test edge cases."""
    xb = 0.5
    var_xb = 0.5
    z = 0.5
    result = pi_wald_interval(xb, var_xb, z)
    assert isinstance(result, dict)
