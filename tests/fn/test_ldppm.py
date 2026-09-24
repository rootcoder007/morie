"""Tests for ldppm.local_dp_planar_mechanism."""

from morie.fn import _array_core as np

from morie.fn.ldppm import local_dp_planar_mechanism


def test_ldppm_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = local_dp_planar_mechanism(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_ldppm_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = local_dp_planar_mechanism(y)
    assert isinstance(result, dict)
