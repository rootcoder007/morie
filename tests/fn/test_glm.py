"""Tests for glm.glr_test."""

import math

from morie.fn import _array_core as np

from morie.fn.glm import glr_test


def test_glm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.integers(0, 2, n)
    p0 = 0.3
    p1 = 0.7
    result = glr_test(x, p0, p1, family="bernoulli")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "estimate" in result
    assert "changepoint" in result
    assert "kl" in result
    assert math.isfinite(result["statistic"])
    assert len(result["estimate"]) == n
    assert math.isfinite(result["kl"])


def test_glm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.normal(0, 1, n)
    p0 = 0.0
    p1 = 1.0
    result = glr_test(x, p0, p1, family="normal", sd=1.0)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "estimate" in result
    assert "changepoint" in result
    assert "kl" in result
    assert math.isfinite(result["statistic"])
    assert len(result["estimate"]) == n
    assert math.isfinite(result["kl"])
