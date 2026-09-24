"""Tests for sd_scale.sd_scale."""

from morie.fn import _array_core as np

from morie.fn.sd_scale import (
    sd_scale,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e41_basic():
    """Test basic functionality."""
    a = 0.5
    sigma = 0.5
    result = sd_scale(a, sigma)
    assert isinstance(result, dict)
    assert "a" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e41_edge():
    """Test edge cases."""
    a = 0.5
    sigma = 0.5
    result = sd_scale(a, sigma)
    assert isinstance(result, dict)
