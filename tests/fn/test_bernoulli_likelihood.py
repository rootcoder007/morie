"""Tests for bernoulli_likelihood.bernoulli_likelihood."""

import math

from morie.fn import _array_core as np

from morie.fn.bernoulli_likelihood import (
    bernoulli_likelihood,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    pis = rng.uniform(0.05, 0.95, 50)
    ys = rng.integers(0, 2, 50)
    result = bernoulli_likelihood(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e1_edge():
    """Test edge cases."""
    pis = [0.3, 0.7, 0.5, 0.2, 0.8, 0.6, 0.4, 0.9]
    ys = [0, 1, 1, 0, 1, 0, 1, 1]
    result = bernoulli_likelihood(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
