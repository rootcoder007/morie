"""Tests for besagl.besag_York_Mollie."""

from morie.fn import _array_core as np

from morie.fn.besagl import besag_York_Mollie


def test_besagl_basic():
    """Test basic functionality."""
    y = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    E = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    A = np.random.default_rng(42).normal(0, 1, (100, 100))
    u = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    v = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = besag_York_Mollie(y, E, A, u, v)
    assert isinstance(result, dict)
    assert "logpost" in result
def test_besagl_edge():
    """Test edge cases."""
    y = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    E = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    A = np.random.default_rng(42).normal(0, 1, (100, 100))
    u = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    v = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = besag_York_Mollie(y, E, A, u, v)
    assert isinstance(result, dict)
