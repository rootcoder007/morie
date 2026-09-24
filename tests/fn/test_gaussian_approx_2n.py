"""Tests for gaussian_approx_2n.gaussian_approx_2n."""

from morie.fn import _array_core as np

import math

from morie.fn.gaussian_approx_2n import (
    gaussian_approx_2n,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e13_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1)  # scalar input
    n = int(rng.integers(10, 100))
    result = gaussian_approx_2n(x, n)
    assert isinstance(result, dict)
    assert "PG" in result
    pg = result["PG"]
    assert math.isfinite(pg)
    assert pg > 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e13_edge():
    """Test edge cases."""
    # Edge case: x = 0, n = 1
    result = gaussian_approx_2n(0, 1)
    assert isinstance(result, dict)
    assert "PG" in result
    pg = result["PG"]
    assert math.isfinite(pg)
    assert pg > 0
