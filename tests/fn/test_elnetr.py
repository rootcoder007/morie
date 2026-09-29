"""Tests for elnetr: the elastic-net KKT conditions (Friedman et al. 2010)."""

import math

import pytest

from morie.fn.elnetr import elnetr

X = [
    [1.0, 0.5, 2.0],
    [2.0, -1.0, 0.1],
    [3.0, 0.2, -1.2],
    [4.0, 1.5, 0.3],
    [5.0, -0.3, 1.1],
    [6.0, 0.8, -0.4],
    [7.0, -1.1, 0.9],
]
Y = [1.1, 2.3, 2.8, 4.4, 4.9, 6.2, 6.8]


def _grad(b0, b, alpha, l1_ratio):
    n = len(Y)
    r = [y - b0 - math.fsum(c * x for c, x in zip(b, row)) for row, y in zip(X, Y)]
    g = [math.fsum(row[j] * ri for row, ri in zip(X, r)) / n - alpha * (1 - l1_ratio) * b[j] for j in range(len(b))]
    return math.fsum(r) / n, g


@pytest.mark.parametrize("alpha,l1_ratio", [(0.1, 0.5), (0.5, 0.3), (2.0, 0.9), (0.05, 0.0)])
def test_elnetr_satisfies_the_kkt_conditions(alpha, l1_ratio):
    """Active coefficients: X_j'r/n - alpha (1 - rho) b_j = alpha rho sign(b_j);
    zero ones: |X_j'r/n| <= alpha rho; intercept: mean residual 0."""
    r = elnetr(X, Y, alpha=alpha, l1_ratio=l1_ratio)
    b = [float(v) for v in r["coef"]]
    rbar, g = _grad(float(r["intercept"]), b, alpha, l1_ratio)
    assert abs(rbar) < 1e-10
    lam = alpha * l1_ratio
    for bj, gj in zip(b, g):
        if bj != 0.0:
            assert gj == pytest.approx(lam * math.copysign(1.0, bj), abs=1e-6)
        else:
            assert abs(gj) <= lam + 1e-6
    assert r["nonzero"] == sum(1 for v in b if abs(v) > 1e-10)


def test_elnetr_ridge_end_matches_the_closed_form():
    """l1_ratio = 0 is ridge: (X_c'X_c + n alpha I) b = X_c'y_c."""
    from morie.fn._qpcore import solve

    n, p, alpha = len(Y), 3, 0.3
    mx = [math.fsum(row[j] for row in X) / n for j in range(p)]
    my = math.fsum(Y) / n
    Xc = [[row[j] - mx[j] for j in range(p)] for row in X]
    A = [[math.fsum(r[a] * r[c] for r in Xc) + (n * alpha if a == c else 0.0) for c in range(p)] for a in range(p)]
    rhs = [math.fsum(r[a] * (y - my) for r, y in zip(Xc, Y)) for a in range(p)]
    b = [float(v) for v in solve(A, rhs)]
    got = elnetr(X, Y, alpha=alpha, l1_ratio=0.0)
    assert [float(v) for v in got["coef"]] == pytest.approx(b, abs=1e-7)
