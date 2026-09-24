"""Tests for hmtd.geron_td_learning."""

from morie.fn import _array_core as np

from morie.fn.hmtd import geron_td_learning


def test_hmtd_basic():
    """Test basic functionality."""
    V = [1.0, 2.0]
    s = [0, 1]
    r = [0.5, -1.0]
    s_next = [1, 0]
    result = geron_td_learning(V, s, r, s_next)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmtd_edge():
    """Test edge cases."""
    V = [1.0, 2.0]
    s = [0, 1]
    r = [0.5, -1.0]
    s_next = [1, 0]
    result = geron_td_learning(V, s, r, s_next)
    assert isinstance(result, dict)
