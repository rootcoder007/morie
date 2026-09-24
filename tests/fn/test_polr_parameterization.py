"""Tests for polr_parameterization.polr_parameterization."""

from morie.fn import _array_core as np

from morie.fn.polr_parameterization import (
    polr_parameterization,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e13_basic():
    """Test basic functionality."""
    bj0 = 0.5
    etas = 0.5
    xs = 0.5
    result = polr_parameterization(bj0, etas, xs)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e13_edge():
    """Test edge cases."""
    bj0 = 0.5
    etas = 0.5
    xs = 0.5
    result = polr_parameterization(bj0, etas, xs)
    assert isinstance(result, dict)
