"""Tests for msm270.mvsml_convolutional_nn_eq_14_7."""

from morie.fn import _array_core as np

from morie.fn.msm270 import mvsml_convolutional_nn_eq_14_7


def test_msm270_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_t = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_7(t, x_t)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm270_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_t = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_7(t, x_t)
    assert isinstance(result, dict)
