"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e2.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2."""

import math

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e2 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    dens = rng.uniform(0.1, 1.0, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2(dens)
    assert hasattr(result, "title")
    assert hasattr(result, "payload")
    assert isinstance(result.payload, dict)
    # Check that the payload contains at least one finite numeric value
    assert any(isinstance(v, (int, float)) and math.isfinite(v) for v in result.payload.values())


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    dens = rng.uniform(0.1, 1.0, 10)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_2(dens)
    assert hasattr(result, "title")
    assert hasattr(result, "payload")
    assert isinstance(result.payload, dict)
    assert any(isinstance(v, (int, float)) and math.isfinite(v) for v in result.payload.values())
