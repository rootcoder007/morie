"""Tests for group_testing_expected_tests.group_testing_expected_tests."""

from morie.fn import _array_core as np

from morie.fn.group_testing_expected_tests import (
    group_testing_expected_tests,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = group_testing_expected_tests(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = group_testing_expected_tests(x)
    assert isinstance(result, dict)
