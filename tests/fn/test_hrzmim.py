"""Tests for hrzmim.multindex: the second-stage smoother and the grid recomputed."""

import math

import pytest

from morie.fn.hrzmim import multindex

N = 60
X = [[-2 + 4 * i / (N - 1), math.cos(0.6 * i)] for i in range(N)]
Z = [r[0] + 0.7 * r[1] for r in X]
Y = [z + 0.3 * z * z for z in Z]


def _k(u):
    return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def test_normalisation_smoother_and_grid():
    r = multindex(X, Y, [[0, 1]], h=0.6, hg=0.3, ngrid=5)
    b = [float(v) for v in r["estimate"][0].tolist()]
    assert b[0] == 1.0
    idx = [float(v) for v in r["indices"][:, 0].tolist()]
    for i in range(N):
        assert abs(idx[i] - (X[i][0] + b[1] * X[i][1])) < 1e-12
    w = [_k((idx[3] - v) / 0.3) for v in idx]
    assert abs(float(r["ghat"][3]) - sum(a * c for a, c in zip(w, Y)) / sum(w)) < 1e-12
    lo, hi = min(idx), max(idx)
    assert r["grid"][0] == lo and abs(r["grid"][-1] - hi) < 1e-12
    g2 = r["grid"][2]
    w = [_k((g2 - v) / 0.3) for v in idx]
    assert abs(r["ggrid"][2] - sum(a * c for a, c in zip(w, Y)) / sum(w)) < 1e-12


def test_grid_needs_single_index():
    with pytest.raises(ValueError):
        multindex(X, Y, [[0], [1]], ngrid=5)
