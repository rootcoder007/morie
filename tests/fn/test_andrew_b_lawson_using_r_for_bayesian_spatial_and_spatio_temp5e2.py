"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2."""

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    y = rng.integers(0, 20, n)
    e = rng.uniform(0.5, 5.0, n)
    n_draws = 100
    theta_draws = rng.normal(0, 0.5, (n_draws, n))
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2(y, e, theta_draws)
    assert isinstance(result, dict)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp5e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    y = rng.integers(0, 20, n)
    e = rng.uniform(0.5, 5.0, n)
    n_draws = 50
    theta_draws = rng.normal(0, 0.5, (n_draws, n))
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_5_equation_2(y, e, theta_draws)
    assert isinstance(result, dict)
