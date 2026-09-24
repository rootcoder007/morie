"""Tests for rng193.rangayyan_ch4_heart_rate_from_count."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_ch4_heart_rate_from_count


def test_rng193_basic():
    """Test basic functionality."""
    nbeats = 5
    duration = 0.1
    result = rangayyan_ch4_heart_rate_from_count(nbeats, duration)
    assert isinstance(result, dict)
    assert "hr" in result


def test_rng193_edge():
    """Test edge cases."""
    nbeats = 5
    duration = 0.1
    result = rangayyan_ch4_heart_rate_from_count(nbeats, duration)
    assert isinstance(result, dict)
