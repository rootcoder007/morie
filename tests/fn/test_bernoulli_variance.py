"""Tests for bernoulli_variance.bernoulli_variance."""

import math

from morie.fn import _array_core as np

from morie.fn.bernoulli_variance import (
    bernoulli_variance,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e22_basic():
    """Test basic functionality."""
    p = 0.3
    result = bernoulli_variance(p)
    assert math.isfinite(result.payload["variance"])
    assert result.payload["p"] == float(p)
    assert math.isclose(result.payload["variance"], p * (1 - p))


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e22_edge():
    """Test edge cases."""
    p = 0.5
    result = bernoulli_variance(p)
    assert math.isfinite(result.payload["variance"])
    assert result.payload["p"] == float(p)
    assert math.isclose(result.payload["variance"], 0.25)
