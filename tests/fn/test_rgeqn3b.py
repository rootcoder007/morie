"""Tests for rgeqn3b.rangayyan_ch3_correlation_sum."""

import math

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch3_correlation_sum


def test_rgeqn3b_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_ch3_correlation_sum(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
    # Verify that all numeric entries returned are finite
    has_numeric = False
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
            has_numeric = True
        elif isinstance(value, str):
            # Some entries may be string labels (e.g. 'R')
            pass
        else:
            # Arrays/lists
            arr = np.asarray(value, dtype=float)
            assert np.all(np.isfinite(arr))
            has_numeric = True
    assert has_numeric


def test_rgeqn3b_edge():
    """Test edge cases."""
    result = rangayyan_ch3_correlation_sum(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
        elif isinstance(value, str):
            pass
        else:
            arr = np.asarray(value, dtype=float)
            assert np.all(np.isfinite(arr))
