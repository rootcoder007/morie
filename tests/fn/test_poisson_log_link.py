"""Tests for poisson_log_link.poisson_log_link."""

from morie.fn import _array_core as np

from morie.fn.poisson_log_link import (
    poisson_log_link,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_log_link(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_log_link(x)
    assert isinstance(result, dict)
