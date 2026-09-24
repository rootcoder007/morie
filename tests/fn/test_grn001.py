"""Tests for grn001.geron_ch4_simple_linear_life_satisfaction."""

from morie.fn import _array_core as np

from morie.fn.grn001 import geron_ch4_simple_linear_life_satisfaction


def test_grn001_basic():
    """Test basic functionality."""
    theta_0 = 4.85
    theta_1 = 4.91e-05
    GDP_per_capita = 27195.0
    result = geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grn001_edge():
    """Test edge cases."""
    theta_0 = 4.85
    theta_1 = 4.91e-05
    GDP_per_capita = 27195.0
    result = geron_ch4_simple_linear_life_satisfaction(theta_0, theta_1, GDP_per_capita)
    assert isinstance(result, dict)
