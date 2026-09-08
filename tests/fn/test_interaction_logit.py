"""Tests for interaction_logit.interaction_logit."""

from morie.fn import _array_core as np

from morie.fn.interaction_logit import (
    interaction_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_logit(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_logit(x)
    assert isinstance(result, dict)
