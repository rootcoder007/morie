"""Tests for jotft.joseph_temporal_fusion_transformer."""

import math

import pytest

from morie.fn.jotft import joseph_temporal_fusion_transformer


def _w(r, c, s):
    return [[math.sin(s + 0.7 * i + 1.3 * j) * 0.5 for j in range(c)] for i in range(r)]


def _ln(v):
    m = sum(v) / len(v)
    return [x - m for x in v]


def test_jotft_basic():
    """Variable-selection weights are a softmax (eq. 6); the prediction is
    W_q grn + b_q (eq. 23); the pinball loss is eq. (25)."""
    a = [0.4, -1.2, 0.9]
    r = joseph_temporal_fusion_transformer(a, _w(3, 3, 0.1), [0.0] * 3, _w(3, 3, 0.2), [0.1] * 3,
                                           _w(3, 3, 0.3), [0.0] * 3, _w(3, 3, 0.4), [0.0] * 3,
                                           _w(3, 3, 0.5), [0.0] * 3, _w(2, 3, 0.6), [0.2, -0.1],
                                           y=[1.0, 0.0], q=0.9)
    assert sum(r["weights"]) == pytest.approx(1.0, rel=1e-14)
    wq = _w(2, 3, 0.6)
    assert r["yhat"] == pytest.approx([sum(wq[i][j] * r["grn"][j] for j in range(3)) + b
                                       for i, b in enumerate([0.2, -0.1])], rel=1e-13)
    ql = sum(0.9 * max(y - h, 0) + 0.1 * max(h - y, 0) for y, h in zip([1.0, 0.0], r["yhat"]))
    assert r["ql"] == pytest.approx(ql / 2, rel=1e-12)     # mean over the two outputs


def test_jotft_edge():
    """A closed GLU gate (eq. 5, bias -50) passes only the skip, so the GRN
    is LayerNorm(a): centred, with unit (population) variance."""
    a = [0.4, -1.2, 0.9]
    r = joseph_temporal_fusion_transformer(a, _w(3, 3, 0.1), [0.0] * 3, _w(3, 3, 0.2), [0.1] * 3,
                                           [[0.0] * 3] * 3, [-50.0] * 3, _w(3, 3, 0.4), [0.0] * 3,
                                           _w(3, 3, 0.5), [0.0] * 3, _w(2, 3, 0.6), [0.0, 0.0])
    assert max(abs(g) for g in r["gate"]) < 1e-20
    c = _ln(a)
    sd = math.sqrt(sum(x * x for x in c) / 3)
    assert r["grn"] == pytest.approx([x / sd for x in c], rel=1e-4)


