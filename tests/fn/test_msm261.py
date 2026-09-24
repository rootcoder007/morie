"""Tests for msm261.mvsml_convolutional_nn_eq_14_1."""

from morie.fn import _array_core as np

from morie.fn.msm261 import mvsml_convolutional_nn_eq_14_1


def test_msm261_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_1(t, x_values, beta_values)
    assert isinstance(result, dict)
    assert "integral" in result


def test_msm261_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta_values = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_1(t, x_values, beta_values)
    assert isinstance(result, dict)
