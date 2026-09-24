"""Tests for lrt_two_groups.lrt_two_groups."""

from morie.fn import _array_core as np

from morie.fn.lrt_two_groups import (
    lrt_two_groups,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e8_basic():
    """Test basic functionality."""
    w1 = 0.5
    n1 = 0.5
    w2 = 0.5
    n2 = 0.5
    result = lrt_two_groups(w1, n1, w2, n2)
    assert isinstance(result, dict)
    assert "stat" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e8_edge():
    """Test edge cases."""
    w1 = 0.5
    n1 = 0.5
    w2 = 0.5
    n2 = 0.5
    result = lrt_two_groups(w1, n1, w2, n2)
    assert isinstance(result, dict)
