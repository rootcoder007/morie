"""Tests for vrmed.variance_based_mediation."""

from morie.fn import _array_core as np

from morie.fn.vrmed import variance_based_mediation


def test_vrmed_basic():
    """Test basic functionality."""
    r2_full = 0.1
    r2_partial = 0.1
    result = variance_based_mediation(r2_full, r2_partial)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vrmed_edge():
    """Test edge cases."""
    r2_full = 0.1
    r2_partial = 0.1
    result = variance_based_mediation(r2_full, r2_partial)
    assert isinstance(result, dict)
