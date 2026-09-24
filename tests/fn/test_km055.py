"""Tests for km055.kamath_ch4_parallel_adapter."""

from morie.fn import _array_core as np

from morie.fn.km055 import kamath_ch4_parallel_adapter


def test_km055_basic():
    """Test basic functionality."""
    H_o = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    H_i = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_down = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    W_up = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = kamath_ch4_parallel_adapter(H_o, H_i, W_down, W_up)
    assert isinstance(result, dict)
    assert "estimate" in result or "output" in result


def test_km055_edge():
    """Test edge cases."""
    H_o = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    H_i = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_down = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    W_up = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = kamath_ch4_parallel_adapter(H_o, H_i, W_down, W_up)
    assert isinstance(result, dict)
