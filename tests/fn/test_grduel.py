"""Tests for grduel.geron_dueling_dqn."""

from morie.fn import _array_core as np

from morie.fn.grduel import geron_dueling_dqn


def test_grduel_basic():
    """Test basic functionality."""
    V = [2.0, -1.0]
    A = [[1.0, 4.0, 1.0], [0.0, 0.0, 3.0]]
    result = geron_dueling_dqn(V, A)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grduel_edge():
    """Test edge cases."""
    V = [2.0, -1.0]
    A = [[1.0, 4.0, 1.0], [0.0, 0.0, 3.0]]
    result = geron_dueling_dqn(V, A)
    assert isinstance(result, dict)
