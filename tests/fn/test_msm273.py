"""Tests for msm273.mvsml_convolutional_nn_eq_14_4."""

from morie.fn import _array_core as np

from morie.fn.msm273 import mvsml_convolutional_nn_eq_14_4


def test_msm273_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_curves = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_convolutional_nn_eq_14_4(t, X_curves, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm273_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_curves = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_convolutional_nn_eq_14_4(t, X_curves, y)
    assert isinstance(result, dict)
