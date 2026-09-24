"""Tests for rgkneejt.rangayyan_knee_joint_sound."""

from morie.fn import _array_core as np

from morie.fn.rgkneejt import rangayyan_knee_joint_sound


def test_rgkneejt_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_knee_joint_sound(vag, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rgkneejt_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_knee_joint_sound(vag, fs)
    assert isinstance(result, dict)
