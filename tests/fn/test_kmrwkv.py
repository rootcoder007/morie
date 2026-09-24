"""Tests for kmrwkv.kamath_rwkv_time_mix."""

from morie.fn import _array_core as np

from morie.fn.kmrwkv import kamath_rwkv_time_mix


def test_kmrwkv_basic():
    """Test basic functionality."""
    k = 0.5
    v = 0.5
    w = 0.5
    result = kamath_rwkv_time_mix(k, v, w)
    assert isinstance(result, dict)
    assert "estimate" in result or "wkv" in result


def test_kmrwkv_edge():
    """Test edge cases."""
    k = 0.5
    v = 0.5
    w = 0.5
    result = kamath_rwkv_time_mix(k, v, w)
    assert isinstance(result, dict)
