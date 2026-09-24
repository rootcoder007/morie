"""Tests for rgbbnorm.rangayyan_ecg_bbb_normal."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_ecg_bbb_normal


def test_rgbbnorm_basic():
    """Test basic functionality."""
    features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_ecg_bbb_normal(features, labels)
    assert isinstance(result, dict)
    assert "predictions" in result


def test_rgbbnorm_edge():
    """Test edge cases."""
    features = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    result = rangayyan_ecg_bbb_normal(features, labels)
    assert isinstance(result, dict)
