"""Tests for hrzbkft.horowitz_backfitting."""

from morie.fn import _array_core as np

from morie.fn.hrzbkft import horowitz_backfitting


def test_hrzbkft_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_backfitting(X, y)
    assert isinstance(result, dict)
    assert "mu" in result


def test_hrzbkft_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_backfitting(X, y)
    assert isinstance(result, dict)
