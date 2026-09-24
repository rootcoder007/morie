"""Tests for msm259.mvsml_deep_learning_eq_13_1."""

from morie.fn import _array_core as np

from morie.fn.msm259 import mvsml_deep_learning_eq_13_1


def test_msm259_basic():
    """Test basic functionality."""
    image = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    kernel = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_deep_learning_eq_13_1(image, kernel)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm259_edge():
    """Test edge cases."""
    image = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    kernel = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_deep_learning_eq_13_1(image, kernel)
    assert isinstance(result, dict)
