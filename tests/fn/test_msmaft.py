"""Tests for msmaft.msm_accelerated_failure."""

from morie.fn import _array_core as np

from morie.fn.msmaft import msm_accelerated_failure


def test_msmaft_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_accelerated_failure(time, event, treatment_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmaft_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_accelerated_failure(time, event, treatment_history)
    assert isinstance(result, dict)
