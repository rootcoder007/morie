"""Tests for km100.kamath_ch6_toxicity_probability."""

from morie.fn import _array_core as np

from morie.fn.km100 import kamath_ch6_toxicity_probability


def test_km100_basic():
    """Test basic functionality."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_toxicity_probability(Yhat, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_km100_edge():
    """Test edge cases."""
    Yhat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_ch6_toxicity_probability(Yhat, c)
    assert isinstance(result, dict)
