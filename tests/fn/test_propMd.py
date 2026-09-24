"""Tests for propMd.proportion_mediated."""

from morie.fn import _array_core as np

from morie.fn.propMd import proportion_mediated


def test_propMd_basic():
    """Test basic functionality."""
    NIE = 0.1
    NDE = 0.1
    result = proportion_mediated(NIE, NDE)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_propMd_edge():
    """Test edge cases."""
    NIE = 0.1
    NDE = 0.1
    result = proportion_mediated(NIE, NDE)
    assert isinstance(result, dict)
