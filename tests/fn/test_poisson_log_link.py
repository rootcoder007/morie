"""Tests for poisson_log_link.poisson_log_link."""

from morie.fn import _array_core as np

from morie.fn.poisson_log_link import (
    poisson_log_link,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e2_basic():
    """Test basic functionality."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    result = poisson_log_link(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e2_edge():
    """Test edge cases."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    result = poisson_log_link(b0, bs, xs)
    assert isinstance(result, dict)
