"""Tests for recapTOA.toa_radiation_balance."""

from morie.fn import _array_core as np

from morie.fn.recapTOA import toa_radiation_balance


def test_recapTOA_basic():
    """Test basic functionality."""
    result = toa_radiation_balance()
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_recapTOA_edge():
    """Test edge cases."""
    result = toa_radiation_balance()
    assert isinstance(result, dict)
