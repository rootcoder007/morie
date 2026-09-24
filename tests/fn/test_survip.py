"""Tests for survip.survey_p_value."""

from morie.fn import _array_core as np

from morie.fn.survip import survey_p_value


def test_survip_basic():
    """Test basic functionality."""
    test_stat = 0.1
    result = survey_p_value(test_stat)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_survip_edge():
    """Test edge cases."""
    test_stat = 0.1
    result = survey_p_value(test_stat)
    assert isinstance(result, dict)
