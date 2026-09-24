"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e9.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_9."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e9 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_9,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    pis = rng.uniform(0.01, 0.99, n)
    ys = rng.integers(0, 2, n)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_9(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], (int, float))
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e9_edge():
    """Test edge cases with small data."""
    rng = np.random.default_rng(123)
    n = 5
    pis = rng.uniform(0.1, 0.9, n)
    ys = rng.integers(0, 2, n)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_9(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], (int, float))
    assert math.isfinite(result["value"])
