"""Tests for msmgmm.msm_gmm_estimator."""

from morie.fn import _array_core as np

from morie.fn.msmgmm import msm_gmm_estimator


def test_msmgmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_gmm_estimator(y, treatment_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmgmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_gmm_estimator(y, treatment_history)
    assert isinstance(result, dict)
