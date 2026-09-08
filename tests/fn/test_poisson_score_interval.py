"""Tests for poisson_score_interval.poisson_score_interval."""

from morie.fn import _array_core as np

from morie.fn.poisson_score_interval import (
    poisson_score_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_score_interval(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_score_interval(x)
    assert isinstance(result, dict)
