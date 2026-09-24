"""Tests for poststs.poststratify."""

import math

from morie.fn import _array_core as np

from morie.fn.poststs import poststratify


def test_poststs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    y = rng.normal(0, 1, n)
    stratum = rng.integers(0, 3, n)  # three strata labelled 0, 1, 2
    Nh = [200, 300, 500]  # positive population sizes, one per stratum
    result = poststratify(y, stratum, Nh)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "variance" in result
    assert "strata" in result
    assert "N" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n
    assert result["strata"] == 3
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    assert result["se"] >= 0


def test_poststs_edge():
    """Test edge cases: two strata with a smaller sample."""
    rng = np.random.default_rng(7)
    n = 40
    y = rng.normal(0, 1, n)
    stratum = rng.integers(0, 2, n)
    Nh = [500, 1500]
    result = poststratify(y, stratum, Nh)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["strata"] == 2
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    assert result["se"] >= 0
