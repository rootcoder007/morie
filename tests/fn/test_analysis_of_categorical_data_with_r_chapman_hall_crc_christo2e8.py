"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e8.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e8 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e8_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    pis = [rng.uniform(0.01, 0.99) for _ in range(n)]
    ys = [int(rng.integers(0, 2)) for _ in range(n)]
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e8_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    n = 5
    pis = [rng.uniform(0.01, 0.99) for _ in range(n)]
    ys = [int(rng.integers(0, 2)) for _ in range(n)]
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_2_equation_8(pis, ys)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
