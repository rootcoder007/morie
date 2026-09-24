"""Tests for binomial_variance.binomial_variance."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.binomial_variance import (
    binomial_variance,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e33_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = int(rng.integers(10, 100))
    p = 0.5
    result = binomial_variance(n, p)
    assert isinstance(result, dict)
    assert "variance" in result.payload
    assert math.isfinite(result.payload["variance"])
    assert result.payload["variance"] == pytest.approx(n * p * (1 - p))


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e33_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = int(rng.integers(10, 100))
    p = 0.0
    result = binomial_variance(n, p)
    assert isinstance(result, dict)
    assert result.payload["variance"] == 0.0
