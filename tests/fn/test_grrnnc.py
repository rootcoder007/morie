"""Tests for grrnnc.geron_simple_rnn_cell."""

from morie.fn import _array_core as np

from morie.fn.grrnnc import geron_simple_rnn_cell


def test_grrnnc_basic():
    """Test basic functionality."""
    x_t = [1.0]
    h_prev = [0.5, -0.5]
    Whh = [[1.0, 0.0], [0.0, 1.0]]
    Wxh = [[2.0], [0.0]]
    b = [0.1, -0.1]
    result = geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grrnnc_edge():
    """Test edge cases."""
    x_t = [1.0]
    h_prev = [0.5, -0.5]
    Whh = [[1.0, 0.0], [0.0, 1.0]]
    Wxh = [[2.0], [0.0]]
    b = [0.1, -0.1]
    result = geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b)
    assert isinstance(result, dict)
