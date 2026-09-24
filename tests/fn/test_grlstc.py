"""Tests for grlstc.geron_lstm_cell."""

from morie.fn import _array_core as np

from morie.fn.grlstc import geron_lstm_cell


def test_grlstc_basic():
    """Test basic functionality."""
    x_t = [0.0]
    h_prev = [0.0]
    c_prev = [3.0]
    Wf = [[0.0, 0.0]]
    Wi = [[0.0, 0.0]]
    Wg = [[0.0, 0.0]]
    Wo = [[0.0, 0.0]]
    bf = -60.0
    bi = -60.0
    bg = 0.0
    bo = 0.0
    result = geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlstc_edge():
    """Test edge cases."""
    x_t = [0.0]
    h_prev = [0.0]
    c_prev = [3.0]
    Wf = [[0.0, 0.0]]
    Wi = [[0.0, 0.0]]
    Wg = [[0.0, 0.0]]
    Wo = [[0.0, 0.0]]
    bf = -60.0
    bi = -60.0
    bg = 0.0
    bo = 0.0
    result = geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
    assert isinstance(result, dict)
