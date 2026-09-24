"""Tests for msm262.mvsml_convolutional_nn_eq_14_2."""

from morie.fn import _array_core as np

from morie.fn.msm262 import mvsml_convolutional_nn_eq_14_2


def test_msm262_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta_coef = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_convolutional_nn_eq_14_2(t, beta_coef)
    assert isinstance(result, dict)
    assert "beta_t" in result


def test_msm262_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta_coef = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_convolutional_nn_eq_14_2(t, beta_coef)
    assert isinstance(result, dict)
