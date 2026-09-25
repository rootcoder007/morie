"""Tests for hmavp.geron_average_pool (Geron ch. 12 average pooling)."""

from morie.fn import _array_core as np

from morie.fn.hmavp import geron_average_pool


def _x():
    # (h, w, c) = (4, 4, 2): channel 0 is 0..15, channel 1 its square
    return [[[float(4 * i + j), float((4 * i + j) ** 2)] for j in range(4)]
            for i in range(4)]


def test_hmavp_basic():
    """2x2 windows, stride 2, valid padding: each output cell is the
    mean of its four inputs, per channel, recomputed here."""
    x = _x()
    result = geron_average_pool(x, 2, 2)
    assert isinstance(result, dict)
    assert result["output_shape"] == (2, 2, 2)
    got = np.asarray(result["pooled"]).tolist()
    for i in range(2):
        for j in range(2):
            for c in range(2):
                win = [x[2 * i + u][2 * j + v][c] for u in (0, 1) for v in (0, 1)]
                assert got[i][j][c] == sum(win) / 4.0
    assert result["parameters"] == 0


def test_hmavp_edge():
    """Overlapping windows (stride 1) on a 3x3 image, and the global pool,
    which is the plain mean over all positions of each channel."""
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = geron_average_pool(a, 2, 1)
    assert np.asarray(result["pooled"]).tolist() == [[3.0, 4.0], [6.0, 7.0]]
    g = geron_average_pool(_x(), global_pool=True)
    got = np.asarray(g["pooled"]).tolist()
    assert got == [sum(range(16)) / 16.0, sum(v * v for v in range(16)) / 16.0]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmavp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
