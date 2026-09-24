"""Tests for qnsc.qn_scale."""

from morie.fn import _array_core as np

from morie.fn.qnsc import qn_scale


def test_qnsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = qn_scale(x)
    assert isinstance(result, dict)
    assert "value" in result


def test_qnsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = qn_scale(x)
    assert isinstance(result, dict)
