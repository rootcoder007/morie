"""Tests for mleth.mle_theta_estimator."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.mleth import mle_theta_estimator


def test_mleth_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    m = 100
    y = rng.integers(0, 2, m).astype(float)
    a = np.ones(m)
    b = rng.normal(0, 1, m)
    result = mle_theta_estimator(y, a, b)
    assert isinstance(result, dict)
    assert "theta" in result
    assert "se" in result
    assert "finite" in result
    assert isinstance(result["finite"], bool)


def test_mleth_edge():
    """Test edge cases - all correct responses yields non-finite estimate."""
    rng = np.random.default_rng(43)
    m = 50
    y = np.ones(m)
    a = np.ones(m)
    b = rng.normal(0, 1, m)
    result = mle_theta_estimator(y, a, b)
    assert isinstance(result, dict)
    assert result["finite"] is False
    assert "why_infinite" in result
    assert isinstance(result["why_infinite"], str)
