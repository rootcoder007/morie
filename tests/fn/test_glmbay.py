"""Tests for glmbay.bayesian_glm."""

from morie.fn import _array_core as np
import pytest

from morie.fn.glmbay import bayesian_glm


def test_glmbay_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    result = bayesian_glm(X, y, family="gaussian", prior_sd=2.5)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_glmbay_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n, p = 40, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = bayesian_glm(X, y, family="binomial", prior_sd=2.5)
    assert isinstance(result, dict)
    assert "estimate" in result
