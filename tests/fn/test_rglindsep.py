"""Tests for bsaclass.rangayyan_lin_discr_sep (Fisher discriminant, sec. 10.4.2)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_lin_discr_sep


def _data():
    X = [[math.sin(0.1 * i) + 0.3 * math.cos(0.37 * i), 0.5 * math.sin(0.1 * i) + 0.2 * math.cos(0.9 * i)]
         for i in range(40)]
    y = [0] * 20 + [1] * 20
    return [[v + (1.5 if c else 0.0) for v in row] for row, c in zip(X, y)], y


def test_rglindsep_basic():
    """w is parallel to S_W^-1 (m1 - m2), with S_W the pooled within-class
    scatter; the midpoint threshold is the mean of the projected class
    means; separable classes give zero training errors."""
    X, y = _data()
    r = rangayyan_lin_discr_sep(X, y)
    m = [[sum(X[i][j] for i in range(40) if y[i] == c) / 20 for j in range(2)] for c in (0, 1)]
    S = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(40):
        d = [X[i][j] - m[y[i]][j] for j in range(2)]
        for a in range(2):
            for b in range(2):
                S[a][b] += d[a] * d[b]
    det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
    Si = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]]
    dm = [m[0][0] - m[1][0], m[0][1] - m[1][1]]
    wref = [Si[0][0] * dm[0] + Si[0][1] * dm[1], Si[1][0] * dm[0] + Si[1][1] * dm[1]]
    w = r["w"]
    assert w[0] * wref[1] - w[1] * wref[0] == pytest.approx(0.0, abs=1e-12)
    pm = [sum(w[0] * X[i][0] + w[1] * X[i][1] for i in range(40) if y[i] == c) / 20 for c in (0, 1)]
    assert r["midpoint_threshold"] == pytest.approx((pm[0] + pm[1]) / 2, rel=1e-12)
    assert r["training_errors"] == 0 and r["training_accuracy"] == 1.0


def test_rglindsep_edge():
    """Fisher's discriminant is two-class: more labels raise."""
    X, _ = _data()
    with pytest.raises(ValueError):
        rangayyan_lin_discr_sep(X, list(range(40)))
