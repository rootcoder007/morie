"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e5.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_5."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e5 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_5,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    b = rng.normal(0, 1, p)
    y = rng.integers(0, 2, n)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_5(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e5_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    b = np.zeros(p)
    y = rng.integers(0, 2, n)
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_5(b, x, y)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
