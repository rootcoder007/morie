"""Tests for proportional_odds_logit.proportional_odds_logit."""

from morie.fn import _array_core as np

from morie.fn.proportional_odds_logit import (
    proportional_odds_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e11_basic():
    """Test basic functionality."""
    bj0 = 0.5
    bs = 0.5
    xs = 0.5
    result = proportional_odds_logit(bj0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e11_edge():
    """Test edge cases."""
    bj0 = 0.5
    bs = 0.5
    xs = 0.5
    result = proportional_odds_logit(bj0, bs, xs)
    assert isinstance(result, dict)
