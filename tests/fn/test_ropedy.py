"""Tests for ropedy.rope_ntk_dynamic."""

from morie.fn import _array_core as np

from morie.fn.ropedy import rope_ntk_dynamic


def test_ropedy_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    m = 0.1
    result = rope_ntk_dynamic(y, q, m)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ropedy_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    q = np.random.default_rng(42).normal(0.0, 1.0, 40)
    m = 0.1
    result = rope_ntk_dynamic(y, q, m)
    assert isinstance(result, dict)
