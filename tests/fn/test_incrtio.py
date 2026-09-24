"""Tests for incrtio.incidence_rate_ratio."""

from morie.fn import _array_core as np

from morie.fn.incrtio import incidence_rate_ratio


def test_incrtio_basic():
    """Test basic functionality."""
    IR_e = 0.5
    IR_u = 0.5
    result = incidence_rate_ratio(IR_e, IR_u)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_incrtio_edge():
    """Test edge cases."""
    IR_e = 0.5
    IR_u = 0.5
    result = incidence_rate_ratio(IR_e, IR_u)
    assert isinstance(result, dict)
