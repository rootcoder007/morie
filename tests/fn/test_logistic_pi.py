"""Tests for logistic_pi.logistic_pi."""

from morie.fn import _array_core as np

from morie.fn.logistic_pi import (
    logistic_pi,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e2_basic():
    """Test basic functionality."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    result = logistic_pi(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e2_edge():
    """Test edge cases."""
    b0 = 0.5
    bs = 0.5
    xs = 0.5
    result = logistic_pi(b0, bs, xs)
    assert isinstance(result, dict)
