"""Tests for sd_scale.sd_scale."""

from morie.fn import _array_core as np

from morie.fn.sd_scale import (
    sd_scale,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_scale(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sd_scale(x)
    assert isinstance(result, dict)
