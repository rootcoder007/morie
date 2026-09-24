"""Tests for nonproportional_odds_logit.nonproportional_odds_logit."""

from morie.fn import _array_core as np

from morie.fn.nonproportional_odds_logit import (
    nonproportional_odds_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e16_basic():
    """Test basic functionality."""
    bj0 = 0.5
    bjs = 0.5
    xs = 0.5
    result = nonproportional_odds_logit(bj0, bjs, xs)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e16_edge():
    """Test edge cases."""
    bj0 = 0.5
    bjs = 0.5
    xs = 0.5
    result = nonproportional_odds_logit(bj0, bjs, xs)
    assert isinstance(result, dict)
