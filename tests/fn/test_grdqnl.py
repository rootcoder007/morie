"""Tests for grdqnl.geron_dqn_loss."""

from morie.fn import _array_core as np

from morie.fn.grdqnl import geron_dqn_loss


def test_grdqnl_basic():
    """Test basic functionality."""
    Q = [[0.0, 0.0], [0.0, 0.0]]
    Q_target = [[0.0, 0.0], [100.0, 0.0]]
    batch = [(0, 0, 2.0, 1, True)]
    result = geron_dqn_loss(Q, Q_target, batch)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdqnl_edge():
    """Test edge cases."""
    Q = [[0.0, 0.0], [0.0, 0.0]]
    Q_target = [[0.0, 0.0], [100.0, 0.0]]
    batch = [(0, 0, 2.0, 1, True)]
    result = geron_dqn_loss(Q, Q_target, batch)
    assert isinstance(result, dict)
