"""Tests for exponential_waiting_density.exponential_waiting_density."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.exponential_waiting_density import (
    exponential_waiting_density,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e26_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    t = float(rng.normal(0, 1, 1)[0])
    lam = 2.0
    result = exponential_waiting_density(t, lam)
    assert isinstance(result, dict)
    assert "density" in result
    density = result["density"]
    expected = lam * math.exp(-lam * t)
    assert math.isclose(density, expected, rel_tol=1e-9)
    assert math.isfinite(density)
    assert result["lambda"] == lam
    assert result["t"] == t


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e26_edge():
    """Test edge cases."""
    t = 0.0
    lam = 1.5
    result = exponential_waiting_density(t, lam)
    assert isinstance(result, dict)
    assert "density" in result
    density = result["density"]
    assert math.isclose(density, lam, rel_tol=1e-9)
    assert math.isfinite(density)
    assert result["lambda"] == lam
    assert result["t"] == t
