"""Tests for grapl.geron_average_pooling (2-D average pooling)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.grapl import geron_average_pooling


def _x():
    return [[float((13 * (7 * i + j)) % 17) for j in range(7)] for i in range(5)]


def _mean(v):
    return sum(v) / len(v)


def test_grapl_basic():
    """Valid padding, 2x3 windows at stride (1, 2): each output is the
    window mean, recomputed; out = ((5-2)//1+1, (7-3)//2+1) = (4, 3)."""
    x = _x()
    result = geron_average_pooling(np.array(x), (2, 3), (1, 2))
    assert isinstance(result, dict)
    got = np.asarray(result["pooled"]).tolist()
    assert result["output_shape"] == (4, 3)
    for i in range(4):
        for j in range(3):
            w = [x[i + a][2 * j + b] for a in range(2) for b in range(3)]
            assert got[i][j] == pytest.approx(_mean(w), rel=1e-15, abs=0)


def test_grapl_edge():
    """SAME padding as tf.nn.avg_pool defines it: ceil(h/s) outputs,
    pad_top = pad_h // 2 (the extra row goes to the bottom), and each
    window averaged over its in-bounds cells only. 3x3 at stride 2 on
    5x7 gives (3, 4) with pad_h = pad_w = 2, so one cell each side."""
    x = _x()
    r = geron_average_pooling(np.array(x), 3, 2, padding="same")
    got = np.asarray(r["pooled"]).tolist()
    assert r["output_shape"] == (3, 4)
    for i in range(3):
        for j in range(4):
            w = [x[a][b] for a in range(2 * i - 1, 2 * i + 2)
                 for b in range(2 * j - 1, 2 * j + 2) if 0 <= a < 5 and 0 <= b < 7]
            assert got[i][j] == pytest.approx(_mean(w), rel=1e-15, abs=0)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.grapl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
