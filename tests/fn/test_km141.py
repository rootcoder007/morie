"""Tests for km141.kamath_ch9_itm_loss."""

import math

from morie.fn import _array_core as np

from morie.fn.km141 import kamath_ch9_itm_loss


def test_km141_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    s = rng.uniform(0.0, 1.0, n)
    y = rng.integers(0, 2, n)
    result = kamath_ch9_itm_loss(s, None, None, y)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "per_pair" in result
    assert "scores" in result
    assert "n" in result
    assert result["n"] == n
    assert math.isfinite(result["estimate"])
    assert len(result["per_pair"]) == n
    assert len(result["scores"]) == n


def test_km141_edge():
    """Test edge cases with small valid input."""
    rng = np.random.default_rng(42)
    n = 3
    s = rng.uniform(0.0, 1.0, n)
    y = rng.integers(0, 2, n)
    result = kamath_ch9_itm_loss(s, None, None, y)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
