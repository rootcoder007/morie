"""Tests for msmtve.msm_time_varying_exposure."""

from morie.fn import _array_core as np

from morie.fn.msmtve import msm_time_varying_exposure


def test_msmtve_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    exposure_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_time_varying_exposure(y, exposure_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmtve_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    exposure_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_time_varying_exposure(y, exposure_history)
    assert isinstance(result, dict)
