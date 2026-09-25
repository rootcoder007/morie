"""Tests for spcar.schabenberger_car_model (Gaussian CAR by ML)."""

import math

import pytest

from morie.fn.spcar import schabenberger_car_model


def _lu_logdet_solve(Q, B):
    """log|Q| and Q^{-1} B by Gaussian elimination with partial pivoting."""
    n = len(Q)
    M = [Q[i][:] + [row[i] for row in B] for i in range(n)]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        ld += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    X = [[0.0] * len(B) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(len(B)):
            X[i][j] = (M[i][n + j] - sum(M[i][k] * X[k][j] for k in range(i + 1, n))) / M[i][i]
    return ld, X


def _grid():
    n = 16
    A = [[1.0 if abs(i % 4 - j % 4) + abs(i // 4 - j // 4) == 1 else 0.0 for j in range(n)] for i in range(n)]
    x = [math.cos(0.7 * k) for k in range(n)]
    z = [1 + 0.5 * x[k] + math.sin(1.3 * k) + 0.3 * math.cos(2.9 * k) for k in range(n)]
    return A, [[1.0, v] for v in x], z


def _profile(rho, A, X, z, form):
    """Concentrated log-likelihood: beta by GLS with precision Q, sigma2 =
    e'Qe/n, l = 1/2 log|Q| - n/2 log sigma2 - n/2 - n/2 log 2pi, with
    Q = D - rho A (weighted, Besag) or I - rho A (identity)."""
    n = len(z)
    d = [sum(r) for r in A]
    Q = [[(d[i] if form == "weighted" else 1.0) * (i == j) - rho * A[i][j] for j in range(n)] for i in range(n)]
    QX = [[sum(Q[i][k] * X[k][c] for k in range(n)) for c in range(2)] for i in range(n)]
    Qz = [sum(Q[i][k] * z[k] for k in range(n)) for i in range(n)]
    XtQX = [[sum(X[i][a] * QX[i][b] for i in range(n)) for b in range(2)] for a in range(2)]
    XtQz = [sum(X[i][a] * Qz[i] for i in range(n)) for a in range(2)]
    det = XtQX[0][0] * XtQX[1][1] - XtQX[0][1] * XtQX[1][0]
    beta = [(XtQX[1][1] * XtQz[0] - XtQX[0][1] * XtQz[1]) / det,
            (XtQX[0][0] * XtQz[1] - XtQX[1][0] * XtQz[0]) / det]
    e = [z[i] - X[i][0] * beta[0] - X[i][1] * beta[1] for i in range(n)]
    s2 = sum(e[i] * Q[i][j] * e[j] for i in range(n) for j in range(n)) / n
    ld, _ = _lu_logdet_solve(Q, [])
    return 0.5 * ld - 0.5 * n * math.log(s2) - 0.5 * n - 0.5 * n * math.log(2 * math.pi), beta, s2


@pytest.mark.parametrize("form", ["weighted", "identity"])
def test_spcar_basic(form):
    """The returned rho maximises the concentrated likelihood (a
    neighbourhood check at +-1e-4 and a vanishing central-difference
    slope), and beta, sigma2 and loglik are the GLS values at that rho.
    The identity form coincides with spatialreg::spautolm(family='CAR')
    on a binary rook grid (rho 0.2340488511, loglik -16.1264482963)."""
    A, X, z = _grid()
    r = schabenberger_car_model(z, A, X, form)
    rho = r.statistic
    ll, beta, s2 = _profile(rho, A, X, z, form)
    assert ll >= _profile(rho + 1e-4, A, X, z, form)[0]
    assert ll >= _profile(rho - 1e-4, A, X, z, form)[0]
    # slope tolerance: the bounded optimiser stops at xatol ~1e-10 times the
    # rho range, and the curvature here is O(10), so |l'| stays below 1e-6
    h = 1e-6
    slope = (_profile(rho + h, A, X, z, form)[0] - _profile(rho - h, A, X, z, form)[0]) / (2 * h)
    assert abs(slope) < 1e-4
    assert [float(b) for b in r.extra["beta"]] == pytest.approx(beta, abs=1e-9)
    assert r.extra["sigma2"] == pytest.approx(s2, rel=1e-9)
    assert r.extra["loglik"] == pytest.approx(ll, abs=1e-9)
    if form == "identity":
        assert rho == pytest.approx(0.2340488511, abs=1e-9)


def test_spcar_edge():
    """Mismatched and asymmetric weights, and unknown forms, raise."""
    A, X, z = _grid()
    with pytest.raises(ValueError):
        schabenberger_car_model(z[:-1], A)
    B = [row[:] for row in A]
    B[0][1] = 2.0
    with pytest.raises(ValueError):
        schabenberger_car_model(z, B)
    with pytest.raises(ValueError):
        schabenberger_car_model(z, A, parameterization="leroux")
