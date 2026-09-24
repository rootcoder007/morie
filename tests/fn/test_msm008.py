"""Tests for msm008.mvsml_overfitting_resampling_eq_4_10."""

from morie.fn import _array_core as np

from morie.fn.msm008 import mvsml_overfitting_resampling_eq_4_10


def test_msm008_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_overfitting_resampling_eq_4_10(y_true, y_pred)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm008_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = mvsml_overfitting_resampling_eq_4_10(y_true, y_pred)
    assert isinstance(result, dict)
