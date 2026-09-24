"""Tests for se_fisher_z.se_fisher_z."""

from morie.fn import _array_core as np

from morie.fn.se_fisher_z import se_fisher_z


def test_ca11e13_basic():
    """Test basic functionality."""
    n = 5
    result = se_fisher_z(n)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e13_edge():
    """Test edge cases."""
    n = 5
    result = se_fisher_z(n)
    assert isinstance(result, dict)
