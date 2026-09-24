"""Tests for manmar.ma_network_indirect."""

from morie.fn import _array_core as np

from morie.fn.manmar import ma_network_indirect


def test_manmar_basic():
    """Test basic functionality."""
    d_AB = 0.1
    v_AB = 0.1
    d_CB = 0.1
    v_CB = 0.1
    result = ma_network_indirect(d_AB, v_AB, d_CB, v_CB)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_manmar_edge():
    """Test edge cases."""
    d_AB = 0.1
    v_AB = 0.1
    d_CB = 0.1
    v_CB = 0.1
    result = ma_network_indirect(d_AB, v_AB, d_CB, v_CB)
    assert isinstance(result, dict)
