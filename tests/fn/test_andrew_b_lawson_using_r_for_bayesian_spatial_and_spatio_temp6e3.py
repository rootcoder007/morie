"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3."""

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    lam0 = rng.uniform(0.1, 1.0, 40)
    lam1 = rng.uniform(0.1, 1.0, 40)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3(
        lam0, lam1
    )
    assert isinstance(result, dict)
    assert len(result) > 0


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    lam0 = rng.uniform(0.5, 2.0, 10)
    lam1 = rng.uniform(0.5, 2.0, 10)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3(
        lam0, lam1
    )
    assert isinstance(result, dict)
    assert len(result) > 0
