"""Tests for wsmscr.wasserman_score_test."""

from morie.fn import _array_core as np

from morie.fn.wsmscr import wasserman_score_test


def test_wsmscr_basic():
    """Test basic functionality."""
    successes = 5
    n = 5
    result = wasserman_score_test(successes, n)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_wsmscr_edge():
    """Test edge cases."""
    successes = 5
    n = 5
    result = wasserman_score_test(successes, n)
    assert isinstance(result, dict)
