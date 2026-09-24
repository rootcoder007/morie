"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e6.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6."""

import math

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e6 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e6_basic():
    """Test basic functionality."""
    t_values = np.linspace(0.0, 4.0, 5)
    counts = np.array([10, 20, 15, 8, 2])
    beta = 0.5
    t_obs = 2.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6(
        t_values, counts, beta, t_obs
    )
    assert isinstance(result, dict)
    assert "p_at_t" in result
    p = float(result["p_at_t"])
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e6_edge():
    """Test edge cases."""
    t_values = np.linspace(0.0, 4.0, 5)
    counts = np.array([10, 20, 15, 8, 2])
    beta = 0.0
    t_obs = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_6(
        t_values, counts, beta, t_obs
    )
    assert isinstance(result, dict)
    assert "p_at_t" in result
    p = float(result["p_at_t"])
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0
