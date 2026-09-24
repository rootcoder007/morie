"""Tests for jotsrg.joseph_ts_as_regression."""

from morie.fn import _array_core as np

from morie.fn.jotsrg import joseph_ts_as_regression


def test_jotsrg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = list(range(1, 11))
    result = joseph_ts_as_regression(y, lags)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_jotsrg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lags = [1]
    result = joseph_ts_as_regression(y, lags)
    assert isinstance(result, dict)
    assert len(result) > 0
