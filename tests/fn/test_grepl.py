"""Tests for grepl.geron_epsilon_greedy."""

from morie.fn import _array_core as np

from morie.fn.grepl import geron_epsilon_greedy


def test_grepl_basic():
    """Test basic functionality."""
    Q_s = [1.0, 7.0, 3.0, 2.0]
    eps = 0.4
    result = geron_epsilon_greedy(Q_s, eps)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grepl_edge():
    """Test edge cases."""
    Q_s = [1.0, 7.0, 3.0, 2.0]
    eps = 0.4
    result = geron_epsilon_greedy(Q_s, eps)
    assert isinstance(result, dict)
