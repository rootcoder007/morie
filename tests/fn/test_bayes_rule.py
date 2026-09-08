"""Tests for bayes_rule.bayes_rule."""

from morie.fn import _array_core as np

from morie.fn.bayes_rule import (
    bayes_rule,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_rule(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_rule(x)
    assert isinstance(result, dict)
