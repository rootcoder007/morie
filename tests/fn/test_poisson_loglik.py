"""Tests for poisson_loglik.poisson_loglik."""

from morie.fn import _array_core as np

from morie.fn.poisson_loglik import (
    poisson_loglik,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e3_basic():
    """Test basic functionality."""
    b = 0.5
    x = 0.5
    y = 0.5
    result = poisson_loglik(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e3_edge():
    """Test edge cases."""
    b = 0.5
    x = 0.5
    y = 0.5
    result = poisson_loglik(b, x, y)
    assert isinstance(result, dict)
