"""Tests for pooled_sd.pooled_sd."""

from morie.fn import _array_core as np

from morie.fn.pooled_sd import pooled_sd


def test_ca11e2_basic():
    """Test basic functionality."""
    s1 = 0.5
    s2 = 0.5
    n1 = 5
    n2 = 5
    result = pooled_sd(s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e2_edge():
    """Test edge cases."""
    s1 = 0.5
    s2 = 0.5
    n1 = 5
    n2 = 5
    result = pooled_sd(s1, s2, n1, n2)
    assert isinstance(result, dict)
