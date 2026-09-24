"""Tests for rgaic.rangayyan_ar_order_aic."""

from morie.fn import _array_core as np

from morie.fn.bsaar import rangayyan_ar_order_aic


def test_rgaic_basic():
    """Test basic functionality."""
    prediction_errors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_samples = 5
    result = rangayyan_ar_order_aic(prediction_errors, n_samples)
    assert isinstance(result, dict)
    assert "order" in result


def test_rgaic_edge():
    """Test edge cases."""
    prediction_errors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n_samples = 5
    result = rangayyan_ar_order_aic(prediction_errors, n_samples)
    assert isinstance(result, dict)
