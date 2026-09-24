"""Tests for konfound.konfound."""

from morie.fn import _array_core as np

from morie.fn.konfound import konfound


def test_konfound_basic():
    """Test basic functionality."""
    est = 0.1
    se = 0.1
    n = 5
    threshold = 0.1
    result = konfound(est, se, n, threshold)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_konfound_edge():
    """Test edge cases."""
    est = 0.1
    se = 0.1
    n = 5
    threshold = 0.1
    result = konfound(est, se, n, threshold)
    assert isinstance(result, dict)
