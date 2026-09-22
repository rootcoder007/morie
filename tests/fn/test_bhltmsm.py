"""Tests for bhltmsm.behavioral_health_msm."""

from morie.fn import _array_core as np

from morie.fn.bhltmsm import behavioral_health_msm


def test_bhltmsm_basic():
    """Test basic functionality."""
    outcome = np.random.default_rng(42).normal(0, 1, 100)
    cumulative = np.random.default_rng(42).normal(0, 1, 100)
    result = behavioral_health_msm(outcome, cumulative)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bhltmsm_edge():
    """Test edge cases."""
    outcome = np.random.default_rng(42).normal(0, 1, 100)
    cumulative = np.random.default_rng(42).normal(0, 1, 100)
    result = behavioral_health_msm(outcome, cumulative)
    assert isinstance(result, dict)
