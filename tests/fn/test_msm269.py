"""Tests for msm269.mvsml_convolutional_nn_eq_14_6."""

from morie.fn import _array_core as np

from morie.fn.msm269 import mvsml_convolutional_nn_eq_14_6


def test_msm269_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_6(t, c)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm269_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    c = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_convolutional_nn_eq_14_6(t, c)
    assert isinstance(result, dict)
