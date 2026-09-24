"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp16e1.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1."""

import math

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp16e1 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp16e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    beta0 = 1.0
    beta1 = 0.5
    x_true = rng.normal(0, 1, n)
    tau = 1.0
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1(beta0, beta1, x_true, tau)
    assert isinstance(result, dict)
    vals = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(vals) > 0
    assert all(math.isfinite(v) for v in vals)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp16e1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    beta0 = 0.0
    beta1 = 0.0
    x_true = rng.normal(0, 1, n)
    tau = 0.5
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_16_equation_1(beta0, beta1, x_true, tau)
    assert isinstance(result, dict)
    vals = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(vals) > 0
    assert all(math.isfinite(v) for v in vals)
