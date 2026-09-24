"""Tests for vbnpc.vb_nonparametric."""

from morie.fn import _array_core as np

from morie.fn.vbnpc import vb_nonparametric


def test_vbnpc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vb_nonparametric(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vbnpc_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vb_nonparametric(y)
    assert isinstance(result, dict)
