"""Tests for grgruc.geron_gru_cell."""

from morie.fn import _array_core as np

from morie.fn.grgruc import geron_gru_cell


def test_grgruc_basic():
    """Test basic functionality."""
    x_t = [1.0]
    h_prev = [0.6]
    Wz = [[0.0, 1.0]]
    Wr = [[0.0, 2.0]]
    W = [[1.0, 0.0]]
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgruc_edge():
    """Test edge cases."""
    x_t = [1.0]
    h_prev = [0.6]
    Wz = [[0.0, 1.0]]
    Wr = [[0.0, 2.0]]
    W = [[1.0, 0.0]]
    result = geron_gru_cell(x_t, h_prev, Wz, Wr, W)
    assert isinstance(result, dict)
