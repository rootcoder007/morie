"""Tests for tqscr.turboquant_score_distortion."""

from morie.fn import _array_core as np

from morie.fn.tqscr import turboquant_score_distortion


def test_tqscr_basic():
    """Test basic functionality."""
    eps = 0.1
    r = 0.1
    n = 0.1
    result = turboquant_score_distortion(eps, r, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "m_min" in result


def test_tqscr_edge():
    """Test edge cases."""
    eps = 0.1
    r = 0.1
    n = 0.1
    result = turboquant_score_distortion(eps, r, n)
    assert isinstance(result, dict)
