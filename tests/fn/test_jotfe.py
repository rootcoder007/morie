"""Tests for jotfe.joseph_calendar_features."""

from morie.fn import _array_core as np

from morie.fn.jotfe import joseph_calendar_features


def test_jotfe_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    years = rng.integers(2000, 2024, n)
    months = rng.integers(1, 13, n)
    days = rng.integers(1, 29, n)
    timestamps = list(zip(years, months, days))
    result = joseph_calendar_features(timestamps)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_jotfe_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 5
    years = rng.integers(2000, 2024, n)
    months = rng.integers(1, 13, n)
    days = rng.integers(1, 29, n)
    timestamps = list(zip(years, months, days))
    result = joseph_calendar_features(timestamps)
    assert isinstance(result, dict)
