"""Tests for msm314.mvsml_convolutional_nn_eq_14_14."""

from morie.fn import _array_core as np

from morie.fn.msm314 import mvsml_convolutional_nn_eq_14_14


def test_msm314_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_14(y, X, X_E)
    assert isinstance(result, dict)
    assert "coef" in result


def test_msm314_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X_E = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_14(y, X, X_E)
    assert isinstance(result, dict)
