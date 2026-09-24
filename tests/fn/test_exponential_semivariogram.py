"""Tests for exponential_semivariogram.exponential_semivariogram."""

import math

from morie.fn import _array_core as np

from morie.fn.exponential_semivariogram import (
    exponential_semivariogram,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e13_basic():
    """Test basic functionality."""
    h = 1.5
    c0, c1, phi = 0.1, 2.0, 3.0
    result = exponential_semivariogram(h, c0, c1, phi)
    assert isinstance(result, dict)
    assert "value" in result
    expected = c0 + c1 * (1 - math.exp(-h / phi))
    assert math.isclose(result["value"], expected)
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e13_edge():
    """Test edge cases: h=0 collapses the exponential term to 0."""
    c0, c1, phi = 0.5, 1.5, 2.0
    result = exponential_semivariogram(0.0, c0, c1, phi)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], 0.0)
    assert math.isfinite(result["value"])
