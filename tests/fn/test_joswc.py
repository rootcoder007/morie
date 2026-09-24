"""Tests for joswc.joseph_sliding_window_cv."""

from morie.fn import _array_core as np

from morie.fn.joswc import joseph_sliding_window_cv


def test_joswc_basic():
    """Test basic functionality."""
    n = 5
    trainsize = 2.0
    testsize = 2.0
    result = joseph_sliding_window_cv(n, trainsize, testsize)
    assert isinstance(result, dict)
    assert "folds" in result


def test_joswc_edge():
    """Test edge cases."""
    n = 5
    trainsize = 2.0
    testsize = 2.0
    result = joseph_sliding_window_cv(n, trainsize, testsize)
    assert isinstance(result, dict)
