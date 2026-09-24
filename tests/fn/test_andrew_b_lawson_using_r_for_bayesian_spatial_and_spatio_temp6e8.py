"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e8.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8."""

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e8 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e8_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0.0, 1.0, 40)
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    R = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8(f, g, R)
    assert isinstance(result, dict)
    assert "eta" in result


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e8_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0.0, 1.0, 40)
    g = np.random.default_rng(42).normal(0.0, 1.0, 40)
    R = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_8(f, g, R)
    assert isinstance(result, dict)
