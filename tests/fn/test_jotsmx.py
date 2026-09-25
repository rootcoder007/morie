"""Tests for jotsmx.joseph_tsmixer."""

import math

import pytest

from morie.fn.jotsmx import joseph_tsmixer


def _w(r, c, s):
    return [[math.sin(s + 0.7 * i + 1.3 * j) * 0.5 for j in range(c)] for i in range(r)]


def _ln(v):
    m = sum(v) / len(v)
    return [x - m for x in v]


def test_jotsmx_basic():
    """Time mixing is shared across channels and feature mixing with
    W = aI + b11' treats channels symmetrically, so permuting the channels
    permutes the forecasts."""
    x = [[1.0, 2.0, 0.5, -1.0], [0.3, -0.2, 0.8, 1.5], [2.0, 1.0, -0.5, 0.0]]
    wt, bt = _w(4, 4, 0.1), [0.05] * 4
    wf = [[0.6 if i == j else 0.2 for j in range(3)] for i in range(3)]
    bf = [0.1] * 3
    wp, bp = _w(2, 4, 0.3), [0.0, 0.1]
    a = joseph_tsmixer(x, wt, bt, wf, bf, wp, bp, 2)
    b = joseph_tsmixer([x[1], x[2], x[0]], wt, bt, wf, bf, wp, bp, 2)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        assert b["forecast"][i] == pytest.approx(a["forecast"][j], rel=1e-12)


def test_jotsmx_edge():
    x = [[1.0, 2.0, 0.5, -1.0]]
    with pytest.raises(ValueError, match="horizon"):
        joseph_tsmixer(x, _w(4, 4, 0.1), [0.0] * 4, [[1.0]], [0.0], _w(3, 4, 0.3), [0.0] * 3, 2)


