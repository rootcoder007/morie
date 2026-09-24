"""Tests for group_testing_expected_tests.group_testing_expected_tests."""

import math

from morie.fn import _array_core as np

from morie.fn.group_testing_expected_tests import (
    group_testing_expected_tests,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e26_basic():
    """Test basic functionality."""
    result = group_testing_expected_tests(10, 0.95, 0.98, 0.01)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e26_edge():
    """Test edge cases."""
    result = group_testing_expected_tests(1, 0.95, 0.98, 0.01)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
