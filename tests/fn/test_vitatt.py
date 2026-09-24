"""Tests for vitatt.vit_self_attention."""

from morie.fn import _array_core as np

from morie.fn.vitatt import vit_self_attention


def test_vitatt_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vit_self_attention(q, k, v)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vitatt_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    k = np.random.default_rng(42).normal(0.0, 1.0, 40)
    v = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vit_self_attention(q, k, v)
    assert isinstance(result, dict)
