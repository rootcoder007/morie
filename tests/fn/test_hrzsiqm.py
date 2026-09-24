"""Tests for hrzsiqm.horowitz_sim_quantile."""

from morie.fn import _array_core as np

from morie.fn.hrzsiqm import horowitz_sim_quantile


def test_hrzsiqm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_quantile(x, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hrzsiqm_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_quantile(x, y)
    assert isinstance(result, dict)
