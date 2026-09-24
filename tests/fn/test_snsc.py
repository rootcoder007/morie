"""Tests for snsc.sn_scale."""

from morie.fn import _array_core as np

from morie.fn.snsc import sn_scale


def test_snsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sn_scale(x)
    assert isinstance(result, dict)
    assert "value" in result


def test_snsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = sn_scale(x)
    assert isinstance(result, dict)
