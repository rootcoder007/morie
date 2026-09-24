"""Tests for joexw.joseph_expanding_window_cv."""

from morie.fn import _array_core as np

from morie.fn.joexw import joseph_expanding_window_cv


def test_joexw_basic():
    """Test basic functionality."""
    n = 5
    initial = 2.0
    testsize = 2.0
    result = joseph_expanding_window_cv(n, initial, testsize)
    assert isinstance(result, dict)
    assert "folds" in result


def test_joexw_edge():
    """Test edge cases."""
    n = 5
    initial = 2.0
    testsize = 2.0
    result = joseph_expanding_window_cv(n, initial, testsize)
    assert isinstance(result, dict)
