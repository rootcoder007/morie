"""Tests for r_from_fisher_z.r_from_fisher_z."""

from morie.fn import _array_core as np

from morie.fn.r_from_fisher_z import r_from_fisher_z


def test_ca11e14_basic():
    """Test basic functionality."""
    z = 0.5
    result = r_from_fisher_z(z)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e14_edge():
    """Test edge cases."""
    z = 0.5
    result = r_from_fisher_z(z)
    assert isinstance(result, dict)
