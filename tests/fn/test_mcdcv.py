"""Tests for mcdcv.min_covariance_determinant."""

import math

from morie.fn import _array_core as np
from morie.fn.mcdcv import min_covariance_determinant


def test_mcdcv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, q = 20, 2
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, q))
    # h must exceed p = q + 1 = 3, and C(20, 8) = 125970 <= 200000
    h = 8
    result = min_covariance_determinant(y, X, h)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "coef" in result
    assert "intercept" in result
    assert result["n"] == n
    assert result["p"] == q + 1
    assert result["h"] == h
    assert len(result["coef"]) == q
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["intercept"])


def test_mcdcv_edge():
    """Test edge cases with a different small valid input size."""
    rng = np.random.default_rng(7)
    n, q = 15, 2
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, q))
    # h must exceed p = q + 1 = 3, and C(15, 5) = 3003 <= 200000
    h = 5
    result = min_covariance_determinant(y, X, h)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "coef" in result
    assert "center" in result
    assert result["n"] == n
    assert result["p"] == q + 1
    assert result["h"] == h
    assert len(result["coef"]) == q
    assert len(result["center"]) == q + 1
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["intercept"])
