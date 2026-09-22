"""Tests for d_from_r_pointbiserial.d_from_r_pointbiserial."""

from morie.fn import _array_core as np

from morie.fn.d_from_r_pointbiserial import d_from_r_pointbiserial


def test_ca11e22_basic():
    """Test basic functionality."""
    r = 0.3
    result = d_from_r_pointbiserial(r)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 2 * r / np.sqrt(1 - r ** 2)
    assert result["value"] == expected


def test_ca11e22_edge():
    """Test edge cases."""
    r = -0.5
    result = d_from_r_pointbiserial(r)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 2 * r / np.sqrt(1 - r ** 2)
    assert result["value"] == expected
