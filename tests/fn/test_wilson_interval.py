"""Tests for wilson_interval.wilson_interval."""

from morie.fn import _array_core as np

from morie.fn.wilson_interval import (
    wilson_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e4_basic():
    """Test basic functionality."""
    w = 0.5
    n = 0.5
    z = 0.5
    result = wilson_interval(w, n, z)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e4_edge():
    """Test edge cases."""
    w = 0.5
    n = 0.5
    z = 0.5
    result = wilson_interval(w, n, z)
    assert isinstance(result, dict)
