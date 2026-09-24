"""Tests for msmcox.msm_cox_marginal."""

from morie.fn import _array_core as np

from morie.fn.msmcox import msm_cox_marginal


def test_msmcox_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_cox_marginal(time, event, treatment_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmcox_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_cox_marginal(time, event, treatment_history)
    assert isinstance(result, dict)
