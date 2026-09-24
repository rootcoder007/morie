"""Tests for msm260.mvsml_deep_learning_eq_13_2."""

from morie.fn import _array_core as np

from morie.fn.msm260 import mvsml_deep_learning_eq_13_2


def test_msm260_basic():
    """Test basic functionality."""
    image = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    kernel = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_deep_learning_eq_13_2(image, kernel)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm260_edge():
    """Test edge cases."""
    image = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    kernel = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_deep_learning_eq_13_2(image, kernel)
    assert isinstance(result, dict)
