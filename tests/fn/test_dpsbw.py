"""Tests for dpsbw.stick_breaking_weights."""

from morie.fn import _array_core as np

from morie.fn.dpsbw import stick_breaking_weights


def test_dpsbw_basic():
    """Test basic functionality."""
    result = stick_breaking_weights()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dpsbw_edge():
    """Test edge cases."""
    result = stick_breaking_weights()
    assert isinstance(result, dict)
