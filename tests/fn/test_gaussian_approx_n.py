"""Tests for gaussian_approx_n.gaussian_approx_n."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.gaussian_approx_n import (
    gaussian_approx_n,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e14_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = float(rng.normal(0, 1))
    n = int(rng.integers(10, 100))
    result = gaussian_approx_n(x, n)
    assert isinstance(result, dict)
    assert "PG" in result
    assert math.isfinite(result["PG"])
    assert result["PG"] > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e14_edge():
    """Test edge cases."""
    x = 0.0
    n = 100
    result = gaussian_approx_n(x, n)
    assert isinstance(result, dict)
    assert "PG" in result
    assert math.isfinite(result["PG"])
    assert result["PG"] > 0
