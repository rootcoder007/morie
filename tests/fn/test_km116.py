"""Tests for km116.kamath_ch8_brevity_penalty."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km116 import kamath_ch8_brevity_penalty


def test_km116_basic():
    """Test basic functionality."""
    c = 3
    r = 5
    result = kamath_ch8_brevity_penalty(c, r)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == pytest.approx(float(np.exp(1.0 - 5 / 3)))
    assert result["c"] == 3.0
    assert result["r"] == 5.0
    assert result["penalized"] is True


def test_km116_edge():
    """Test edge cases."""
    c = 7
    r = 5
    result = kamath_ch8_brevity_penalty(c, r)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["estimate"] == 1.0
    assert result["penalized"] is False
    # also check the c == r boundary
    out = kamath_ch8_brevity_penalty(5, 5)
    assert out["estimate"] == 1.0
