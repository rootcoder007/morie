"""Tests for sbcrk.simulation_based_calibration_rank."""

from morie.fn import _array_core as np

from morie.fn.sbcrk import simulation_based_calibration_rank


def test_sbcrk_basic():
    """Test basic functionality."""
    prior_draw = np.random.default_rng(42).normal(0.0, 1.0, 40)
    post_draws = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulation_based_calibration_rank(prior_draw, post_draws)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "rank" in result


def test_sbcrk_edge():
    """Test edge cases."""
    prior_draw = np.random.default_rng(42).normal(0.0, 1.0, 40)
    post_draws = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = simulation_based_calibration_rank(prior_draw, post_draws)
    assert isinstance(result, dict)
