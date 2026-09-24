"""Tests for msm249.mvsml_reproducing_kernel_eq_10_5."""

from morie.fn import _array_core as np

from morie.fn.msm249 import mvsml_reproducing_kernel_eq_10_5


def test_msm249_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_5(y_hat, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm249_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_reproducing_kernel_eq_10_5(y_hat, y)
    assert isinstance(result, dict)
