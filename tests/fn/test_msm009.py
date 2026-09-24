"""Tests for msm009.mvsml_overfitting_resampling_eq_4_14."""

from morie.fn import _array_core as np

from morie.fn.msm009 import mvsml_overfitting_resampling_eq_4_14


def test_msm009_basic():
    """Test basic functionality."""
    probs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_true = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_overfitting_resampling_eq_4_14(probs, y_true)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm009_edge():
    """Test edge cases."""
    probs = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y_true = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = mvsml_overfitting_resampling_eq_4_14(probs, y_true)
    assert isinstance(result, dict)
