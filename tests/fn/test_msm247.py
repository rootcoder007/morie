"""Tests for msm247.mvsml_reproducing_kernel_eq_10_6."""

from morie.fn import _array_core as np

from morie.fn.msm247 import mvsml_reproducing_kernel_eq_10_6


def test_msm247_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_6(X, W_h)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm247_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    W_h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_6(X, W_h)
    assert isinstance(result, dict)
