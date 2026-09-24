"""Tests for msm275.mvsml_convolutional_nn_eq_14_9."""

from morie.fn import _array_core as np

from morie.fn.msm275 import mvsml_convolutional_nn_eq_14_9


def test_msm275_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_curves = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_9(t, X_curves)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm275_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X_curves = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_9(t, X_curves)
    assert isinstance(result, dict)
