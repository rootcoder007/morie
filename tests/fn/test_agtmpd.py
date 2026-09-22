"""Tests for agtmpd.alphazero_temp_decay."""

from morie.fn import _array_core as np

from morie.fn.agtmpd import alphazero_temp_decay


def test_agtmpd_basic():
    """Test basic functionality."""
    move_count = 100
    result = alphazero_temp_decay(move_count)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_agtmpd_edge():
    """Test edge cases."""
    move_count = 100
    result = alphazero_temp_decay(move_count)
    assert isinstance(result, dict)
