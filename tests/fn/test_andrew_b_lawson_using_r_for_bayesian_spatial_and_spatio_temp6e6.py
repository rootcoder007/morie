"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e6.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e6 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    eta = rng.normal(0, 1, n)
    y = rng.integers(0, 2, n)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6(eta, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(numeric_values) > 0
    assert all(math.isfinite(v) for v in numeric_values)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e6_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 10
    eta = rng.normal(0, 1, n)
    y = rng.integers(0, 2, n)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_6(eta, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    numeric_values = [v for v in result.values() if isinstance(v, (int, float))]
    assert len(numeric_values) > 0
    assert all(math.isfinite(v) for v in numeric_values)
