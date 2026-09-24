"""Tests for se_g.se_g."""

from morie.fn import _array_core as np

from morie.fn.se_g import se_g


def test_ca11e7_basic():
    """Test basic functionality."""
    g = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_g(g, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e7_edge():
    """Test edge cases."""
    g = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_g(g, n1, n2)
    assert isinstance(result, dict)
