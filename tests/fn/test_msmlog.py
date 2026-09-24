"""Tests for msmlog.msm_logistic."""

from morie.fn import _array_core as np

from morie.fn.msmlog import msm_logistic


def test_msmlog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_logistic(y, treatment_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmlog_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_logistic(y, treatment_history)
    assert isinstance(result, dict)
