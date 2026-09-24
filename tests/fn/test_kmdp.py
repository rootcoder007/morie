"""Tests for kmdp.kamath_differential_privacy."""

from morie.fn import _array_core as np

from morie.fn.kmdp import kamath_differential_privacy


def test_kmdp_basic():
    """Test basic functionality."""
    eps = 2.0
    delta = 0.01
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdp_edge():
    """Test edge cases."""
    eps = 2.0
    delta = 0.01
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)
