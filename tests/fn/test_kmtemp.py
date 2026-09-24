"""Tests for kmtemp.kamath_temperature_sampling."""

from morie.fn import _array_core as np

from morie.fn.kmtemp import kamath_temperature_sampling


def test_kmtemp_basic():
    """Test basic functionality."""
    logits = 0.5
    T = 0.5
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)
    assert "estimate" in result or "probabilities" in result


def test_kmtemp_edge():
    """Test edge cases."""
    logits = 0.5
    T = 0.5
    result = kamath_temperature_sampling(logits, T)
    assert isinstance(result, dict)
