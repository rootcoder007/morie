"""Tests for spsarml.schabenberger_sar_ml (SAR error / lag models by ML)."""

import math

import pytest

from morie.fn.spsarml import schabenberger_sar_ml


def _inv(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def _logdet(A):
    n = len(A)
    M = [row[:] for row in A]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        ld += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return ld


def mm(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _setup(g=4):
    n = g * g
    A = [[1.0 if abs(i % g - j % g) + abs(i // g - j // g) == 1 else 0.0 for j in range(n)] for i in range(n)]
    W = [[v / sum(r) for v in r] for r in A]
    k = range(n)
    x = [math.cos(0.9 * t) + 0.3 * math.sin(2.1 * t) for t in k]
    e0 = [0.4 * math.sin(5.7 * t + 1) for t in k]
    B = [[(1.0 if i == j else 0.0) - 0.4 * W[i][j] for j in range(n)] for i in range(n)]
    Bi = _inv(B)
    y = [sum(Bi[i][j] * (1 + 2 * x[j] + e0[j]) for j in range(n)) for i in range(n)]
    X = [[1.0, v] for v in x]
    return X, y, W


def _fit(X, y, W, rho, model):
    """Concentrated -2 log L(rho) = n log(2 pi s2) + n - 2 log|I - rho W|
    with beta, s2 by least squares on the transformed model."""
    n = len(y)
    A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
    if model == "error":
        Xs = mm(A, X)
        ys = [sum(A[i][j] * y[j] for j in range(n)) for i in range(n)]
    else:
        Xs = X
        ys = [sum(A[i][j] * y[j] for j in range(n)) for i in range(n)]
    XtX = mm(list(map(list, zip(*Xs))), Xs)
    b = [sum(r) for r in mm(_inv(XtX), [[sum(Xs[i][a] * ys[i] for i in range(n))] for a in range(2)])]
    r = [ys[i] - sum(Xs[i][a] * b[a] for a in range(2)) for i in range(n)]
    s2 = sum(t * t for t in r) / n
    return n * math.log(2 * math.pi * s2) + n - 2 * _logdet(A), b, s2, A


@pytest.mark.parametrize("model", ["error", "lag"])
def test_spsarml_basic(model):
    """rho maximises the concentrated likelihood (recomputed here);
    beta and sigma^2 are the profiled values; the standard errors are
    the analytic information matrix of Ord (1975) / Anselin (1988),
    rebuilt here from W A^-1.  (On a 5x5 version of this design the
    estimates and every standard error matched spatialreg errorsarlm /
    lagsarlm, method = "eigen", to 1e-7.)"""
    X, y, W = _setup()
    n = len(y)
    r = schabenberger_sar_ml(X, y, W, model)
    rho = r["rho"]
    f0, b, s2, A = _fit(X, y, W, rho, model)
    # golden-section stops at a bracket below 1e-12; the objective is
    # flat to second order there, so neighbours at +-1e-5 are not lower
    assert f0 <= _fit(X, y, W, rho + 1e-5, model)[0] + 1e-12
    assert f0 <= _fit(X, y, W, rho - 1e-5, model)[0] + 1e-12
    assert r["neg2loglik"] == pytest.approx(f0, rel=1e-12)
    assert [float(v) for v in r["beta"]] == pytest.approx(b, abs=1e-10)
    assert r["sigma2"] == pytest.approx(s2, rel=1e-10)
    WA = mm(W, _inv(A))
    tr1 = sum(WA[i][i] for i in range(n))
    WA2 = mm(WA, WA)
    t2 = sum(WA2[i][i] for i in range(n)) + sum(WA[i][j] ** 2 for i in range(n) for j in range(n))
    if model == "error":
        Xe = mm(A, X)
        cb = [[s2 * v for v in row] for row in _inv(mm(list(map(list, zip(*Xe))), Xe))]
        I2 = [[n / (2 * s2 * s2), tr1 / s2], [tr1 / s2, t2]]
        vr = _inv(I2)[1][1]
    else:
        Xb = [sum(X[i][a] * b[a] for a in range(2)) for i in range(n)]
        WXb = [sum(WA[i][j] * Xb[j] for j in range(n)) for i in range(n)]
        I = [[0.0] * 4 for _ in range(4)]
        for a in range(2):
            for c in range(2):
                I[a][c] = sum(X[i][a] * X[i][c] for i in range(n)) / s2
            I[a][3] = I[3][a] = sum(X[i][a] * WXb[i] for i in range(n)) / s2
        I[2][2] = n / (2 * s2 * s2)
        I[2][3] = I[3][2] = tr1 / s2
        I[3][3] = t2 + sum(v * v for v in WXb) / s2
        Ii = _inv(I)
        cb = [row[:2] for row in Ii[:2]]
        vr = Ii[3][3]
    assert [float(v) for v in r["se"]] == pytest.approx([math.sqrt(cb[0][0]), math.sqrt(cb[1][1])], rel=1e-8)
    assert r["se_rho"] == pytest.approx(math.sqrt(vr), rel=1e-8)


def test_spsarml_edge():
    """The least-squares rho is the regression of the OLS residuals on
    their lag (error model); a non-square W, a non-zero diagonal and an
    unknown model raise."""
    X, y, W = _setup()
    n = len(y)
    r = schabenberger_sar_ml(X, y, W, "error")
    XtX = mm(list(map(list, zip(*X))), X)
    bo = [sum(v) for v in mm(_inv(XtX), [[sum(X[i][a] * y[i] for i in range(n))] for a in range(2)])]
    e = [y[i] - bo[0] - bo[1] * X[i][1] for i in range(n)]
    We = [sum(W[i][j] * e[j] for j in range(n)) for i in range(n)]
    assert r["ols_rho"] == pytest.approx(sum(a * b for a, b in zip(We, e)) / sum(a * a for a in We), rel=1e-9)
    assert r["row_standardised"] is True
    with pytest.raises(ValueError):
        schabenberger_sar_ml(X, y, [row[:-1] for row in W])
    Wd = [row[:] for row in W]
    Wd[0][0] = 0.5
    with pytest.raises(ValueError):
        schabenberger_sar_ml(X, y, Wd)
    with pytest.raises(ValueError):
        schabenberger_sar_ml(X, y, W, "durbin")


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.spsarml as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
