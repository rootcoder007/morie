"""Tests for eqmm.equating_mean_mean."""

from morie.fn import _array_core as np

from morie.fn.eqmm import equating_mean_mean


def test_eqmm_basic():
    """Test basic functionality."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    a_R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_F = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_F = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_mean_mean(y, a_R, b_R, a_F, b_F)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_eqmm_edge():
    """Test edge cases."""
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    a_R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_R = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_F = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_F = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_mean_mean(y, a_R, b_R, a_F, b_F)
    assert isinstance(result, dict)
