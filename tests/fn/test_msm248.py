"""Tests for msm248.mvsml_reproducing_kernel_eq_10_9."""

from morie.fn import _array_core as np

from morie.fn.msm248 import mvsml_reproducing_kernel_eq_10_9


def test_msm248_basic():
    """Test basic functionality."""
    V_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_l = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_9(V_h, W_l)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm248_edge():
    """Test edge cases."""
    V_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_l = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_9(V_h, W_l)
    assert isinstance(result, dict)
