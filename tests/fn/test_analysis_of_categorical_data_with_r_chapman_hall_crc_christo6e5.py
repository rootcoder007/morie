"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e5.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5."""

from morie.fn import _array_core as np

import math

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e5 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    t_values = np.linspace(0.0, 5.0, 10)
    counts = rng.integers(1, 100, size=10)
    beta = 0.5
    t_obs = float(t_values[5])
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5(t_values, counts, beta, t_obs)
    assert isinstance(result, dict)
    assert 'p_at_t' in result
    p = result['p_at_t']
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e5_edge():
    """Test edge cases."""
    # Minimal valid input
    t_values = np.array([0.0, 1.0])
    counts = np.array([1, 1], dtype=int)
    beta = 1.0
    t_obs = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_5(t_values, counts, beta, t_obs)
    assert isinstance(result, dict)
    assert 'p_at_t' in result
    p = result['p_at_t']
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
