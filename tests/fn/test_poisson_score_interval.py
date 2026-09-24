"""Tests for poisson_score_interval.poisson_score_interval."""

from morie.fn import _array_core as np

from morie.fn.poisson_score_interval import (
    poisson_score_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e1_basic():
    """Test basic functionality."""
    mu_hat = 0.5
    n = 0.5
    z = 0.5
    result = poisson_score_interval(mu_hat, n, z)
    assert isinstance(result, dict)
    assert "lower" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e1_edge():
    """Test edge cases."""
    mu_hat = 0.5
    n = 0.5
    z = 0.5
    result = poisson_score_interval(mu_hat, n, z)
    assert isinstance(result, dict)
