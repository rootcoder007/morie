"""Tests for sd_of_mean_hetero.sd_of_mean_hetero."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.sd_of_mean_hetero import (
    sd_of_mean_hetero,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e55_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    sigmas = np.abs(rng.normal(0, 1, 10))
    result = sd_of_mean_hetero(sigmas)
    assert isinstance(result, dict)
    assert "sd_avg" in result
    assert math.isfinite(result["sd_avg"])
    assert result["sd_avg"] >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e55_edge():
    """Test edge cases."""
    sigmas = [0.5, 0.5, 0.5]
    result = sd_of_mean_hetero(sigmas)
    assert isinstance(result, dict)
    assert "sd_avg" in result
    assert math.isfinite(result["sd_avg"])
    assert result["sd_avg"] >= 0.0
