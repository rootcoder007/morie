"""Tests for gaussian_sum_density.gaussian_sum_density."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.gaussian_sum_density import (
    gaussian_sum_density,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e70_basic():
    """Test basic functionality."""
    z = 0.0
    sigma_x = 1.0
    sigma_y = 2.0
    result = gaussian_sum_density(z, sigma_x, sigma_y)
    # The function returns a RichResult that behaves like a dict.
    assert isinstance(result, dict)
    # The payload contains 'density' and 'sigma_sum'.
    assert "density" in result
    assert "sigma_sum" in result
    # Both should be finite numbers.
    assert math.isfinite(result["density"])
    assert math.isfinite(result["sigma_sum"])
    # sigma_sum should equal sqrt(sigma_x^2 + sigma_y^2).
    expected_sigma_sum = math.sqrt(sigma_x ** 2 + sigma_y ** 2)
    assert result["sigma_sum"] == pytest.approx(expected_sigma_sum)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e70_edge():
    """Test edge cases."""
    # Edge case: small but positive sigmas with a large |z|, still valid.
    z = -5.0
    sigma_x = 0.5
    sigma_y = 0.5
    result = gaussian_sum_density(z, sigma_x, sigma_y)
    assert isinstance(result, dict)
    assert "density" in result
    assert "sigma_sum" in result
    assert math.isfinite(result["density"])
    # sigma_sum should still be sqrt(0.25 + 0.25).
    expected_sigma_sum = math.sqrt(sigma_x ** 2 + sigma_y ** 2)
    assert result["sigma_sum"] == pytest.approx(expected_sigma_sum)
