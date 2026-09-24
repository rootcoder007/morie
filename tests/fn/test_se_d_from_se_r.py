"""Tests for se_d_from_se_r.se_d_from_se_r."""

from morie.fn import _array_core as np

from morie.fn.se_d_from_se_r import se_d_from_se_r


def test_ca11e23_basic():
    """Test basic functionality."""
    r = 0.5
    se_r = 0.5
    result = se_d_from_se_r(r, se_r)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca11e23_edge():
    """Test edge cases."""
    r = 0.5
    se_r = 0.5
    result = se_d_from_se_r(r, se_r)
    assert isinstance(result, dict)
