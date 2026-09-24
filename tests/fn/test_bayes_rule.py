"""Tests for bayes_rule.bayes_rule."""

import math

from morie.fn.bayes_rule import (
    bayes_rule,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e22_basic():
    """Test basic functionality."""
    p_a_given_b = 0.8
    p_b = 0.4
    p_a_given_notb = 0.1
    result = bayes_rule(p_a_given_b, p_b, p_a_given_notb)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    value = result["value"]
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e22_edge():
    """Test edge cases."""
    # When P(A|B) == P(A|~B), Bayes rule should return P(B) unchanged.
    p_a_given_b = 0.5
    p_b = 0.3
    p_a_given_notb = 0.5
    result = bayes_rule(p_a_given_b, p_b, p_a_given_notb)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
    assert abs(value - p_b) < 1e-12
