"""Tests for grpels.geron_peephole_lstm_cell."""

from morie.fn import _array_core as np

from morie.fn.grpels import geron_peephole_lstm_cell


def test_grpels_basic():
    """Test basic functionality."""
    x_t = [0.0]
    h_prev = [0.0]
    c_prev = [1.0]
    Wf = [[0.0, 0.0]]
    Wi = [[0.0, 0.0]]
    Wg = [[0.0, 0.0]]
    Wo = [[0.0, 0.0]]
    Uf = [2.0]
    Ui = [0.0]
    Uo = [0.0]
    bf = [0.0]
    bi = [0.0]
    bg = [0.0]
    bo = [0.0]
    result = geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grpels_edge():
    """Test edge cases."""
    x_t = [0.0]
    h_prev = [0.0]
    c_prev = [1.0]
    Wf = [[0.0, 0.0]]
    Wi = [[0.0, 0.0]]
    Wg = [[0.0, 0.0]]
    Wo = [[0.0, 0.0]]
    Uf = [2.0]
    Ui = [0.0]
    Uo = [0.0]
    bf = [0.0]
    bi = [0.0]
    bg = [0.0]
    bo = [0.0]
    result = geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo)
    assert isinstance(result, dict)
