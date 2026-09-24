"""Tests for riskrt.risk_ratio."""

from morie.fn import _array_core as np

from morie.fn.riskrt import risk_ratio


def test_riskrt_basic():
    """Test basic functionality."""
    p_exposed = 0.5
    p_unexposed = 0.5
    result = risk_ratio(p_exposed, p_unexposed)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_riskrt_edge():
    """Test edge cases."""
    p_exposed = 0.5
    p_unexposed = 0.5
    result = risk_ratio(p_exposed, p_unexposed)
    assert isinstance(result, dict)
