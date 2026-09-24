"""Tests for jzdiff.jenson_zhang_disparity."""

import math

from morie.fn import _array_core as np

from morie.fn.jzdiff import jenson_zhang_disparity


def test_jzdiff_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 100
    y = np.arange(n)
    p = np.abs(rng.normal(0, 1, n))
    q = np.abs(rng.normal(0, 1, n))
    result = jenson_zhang_disparity(y, p, q)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "divergence" in result
    assert "distance" in result
    assert "entropy_p" in result
    assert "entropy_q" in result
    assert "entropy_m" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0
    assert result["divergence"] >= 0
    assert math.isfinite(result["distance"])
    assert result["distance"] >= 0


def test_jzdiff_edge():
    """Test edge case: identical distributions yield zero divergence."""
    rng = np.random.default_rng(43)
    n = 50
    y = np.arange(n)
    p = np.abs(rng.normal(0, 1, n))
    q = list(p)
    result = jenson_zhang_disparity(y, p, q)
    assert isinstance(result, dict)
    assert result["estimate"] == 0.0
    assert result["divergence"] == 0.0
    assert result["distance"] == 0.0
    assert result["entropy_p"] == result["entropy_q"]
    assert result["n"] == n
