"""Tests for or_wald_interval.or_wald_interval."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.or_wald_interval import (
    or_wald_interval,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e10_basic():
    """Test basic functionality."""
    result = or_wald_interval(10, 50, 20, 60, 1.96)
    assert isinstance(result, dict)
    assert "or" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e10_edge():
    """Test edge cases."""
    result = or_wald_interval(1, 5, 4, 5, 1.96)
    assert isinstance(result, dict)
    assert "or" in result
    assert math.isfinite(result["value"])
