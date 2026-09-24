"""Tests for se_d_probit.se_d_probit."""

from morie.fn import _array_core as np

from morie.fn.se_d_probit import se_d_probit


def test_ca11e21_basic():
    """Test basic functionality."""
    p1 = 0.5
    p2 = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_d_probit(p1, p2, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca11e21_edge():
    """Test edge cases."""
    p1 = 0.5
    p2 = 0.5
    n1 = 0.5
    n2 = 0.5
    result = se_d_probit(p1, p2, n1, n2)
    assert isinstance(result, dict)
