"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e1.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e1 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1,
)


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    # Build a valid PMF: non-negative values normalised to sum to 1.
    raw = rng.uniform(0, 1, n)
    total = sum(raw)
    dens = [d / total for d in raw]

    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1(dens)

    # The function returns a dict-like result; extract the numeric joint likelihood.
    assert isinstance(result, dict)
    numeric_values = [
        v for v in result.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert len(numeric_values) >= 1
    payload = float(numeric_values[0])
    assert math.isfinite(payload)
    # A product of probabilities in [0, 1] is itself in [0, 1].
    assert 0.0 <= payload <= 1.0


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp3e1_edge():
    """Test edge cases."""
    # A small but valid uniform PMF.
    dens = [0.2, 0.2, 0.2, 0.2, 0.2]

    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_3_equation_1(dens)

    assert isinstance(result, dict)
    numeric_values = [
        v for v in result.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert len(numeric_values) >= 1
    payload = float(numeric_values[0])
    assert math.isfinite(payload)
    assert 0.0 <= payload <= 1.0
