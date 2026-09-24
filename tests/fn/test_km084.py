"""Tests for km084.kamath_ch6_lpbs_bias."""

from morie.fn import _array_core as np
import math

from morie.fn.km084 import kamath_ch6_lpbs_bias


def test_km084_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p_a = rng.uniform(0.01, 1, 2)
    p_prior = rng.uniform(0.01, 1, 2)
    result = kamath_ch6_lpbs_bias(p_a, p_prior)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 2


def test_km084_edge():
    """Test edge cases."""
    # Equal target and prior probabilities yield zero bias.
    p_a = [0.5, 0.5]
    p_prior = [0.5, 0.5]
    result = kamath_ch6_lpbs_bias(p_a, p_prior)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 0.0
    assert result["n"] == 2
