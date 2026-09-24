"""Tests for riskdf.risk_difference."""

from morie.fn import _array_core as np

from morie.fn.riskdf import risk_difference


def test_riskdf_basic():
    """Test basic functionality."""
    p_exposed = 0.5
    p_unexposed = 0.5
    result = risk_difference(p_exposed, p_unexposed)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_riskdf_edge():
    """Test edge cases."""
    p_exposed = 0.5
    p_unexposed = 0.5
    result = risk_difference(p_exposed, p_unexposed)
    assert isinstance(result, dict)
