"""Tests for jntent.joint_entropy."""

from morie.fn import _array_core as np

from morie.fn.jntent import joint_entropy


def test_jntent_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joint_entropy(pxy)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_jntent_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joint_entropy(pxy)
    assert isinstance(result, dict)
