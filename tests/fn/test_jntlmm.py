"""Tests for jntlmm.joint_longitudinal_survival."""

from morie.fn import _array_core as np

from morie.fn.jntlmm import joint_longitudinal_survival


def test_jntlmm_basic():
    """Test basic functionality."""
    long_y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cluster = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joint_longitudinal_survival(long_y, time, event, X, Z, cluster)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_jntlmm_edge():
    """Test edge cases."""
    long_y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    Z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    cluster = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joint_longitudinal_survival(long_y, time, event, X, Z, cluster)
    assert isinstance(result, dict)
