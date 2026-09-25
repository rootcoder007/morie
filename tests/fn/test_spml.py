"""Tests for spml.schabenberger_ml_variogram (ML / REML, eqs 4.35-4.40)."""

import math

import pytest

from morie.fn.spml import schabenberger_ml_variogram


def _data(n=14):
    x = [10 * ((k * 0.618034) % 1) for k in range(n)]
    y = [10 * ((k * 0.414214 + 0.3) % 1) for k in range(n)]
    e = [((math.sin(12.9898 * k + 78.233) * 43758.5453) % 1) - 0.5 for k in range(n)]
    z = [sum(math.exp(-math.dist((x[i], y[i]), (x[j], y[j])) / 1.5) * e[j] for j in range(n))
         + 0.6 * (((math.sin(3.3 * i) * 1000) % 1) - 0.5) for i in range(n)]
    return [[a, b] for a, b in zip(x, y)], z


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return L


def _solve_chol(L, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _m2ll(co, z, nug, ps, a, reml):
    """-2 log L with the practical-range exponential covariance
    C(h) = ps exp(-3h/a), C(0) = nug + ps, and the constant mean profiled
    by GLS; REML adds log(1' S^-1 1) and uses n - 1 in the 2 pi term."""
    n = len(z)
    S = [[(nug + ps) if i == j else ps * math.exp(-3 * math.dist(co[i], co[j]) / a)
          for j in range(n)] for i in range(n)]
    L = _chol(S)
    s1 = _solve_chol(L, [1.0] * n)
    sz = _solve_chol(L, z)
    xtx = sum(s1)
    mu = sum(sz) / xtx
    r = [t - mu for t in z]
    quad = sum(a_ * b_ for a_, b_ in zip(r, _solve_chol(L, r)))
    ld = 2 * sum(math.log(L[i][i]) for i in range(n))
    if reml:
        return ld + math.log(xtx) + quad + (n - 1) * math.log(2 * math.pi), mu
    return ld + quad + n * math.log(2 * math.pi), mu


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_spml_basic(method):
    """The reported -2 log L is the objective recomputed independently at
    the returned parameters, the GLS mean matches, and no perturbation of
    one log-parameter by +-1e-3 lowers the objective (a local optimum).
    The 1e-7 tolerances cover the module's 1e-10 * mean-variance diagonal
    jitter, amplified at most by n/lambda_min(S) with lambda_min above
    the fitted nugget.  On a 25-site version of this design the ML fit
    was checked against nlme::gls(corExp(nugget = TRUE)): same range and
    partial sill to 1e-4, with morie reaching the lower -2 log L."""
    co, z = _data()
    r = schabenberger_ml_variogram(co, z, "exponential", method)
    reml = method == "reml"
    v, mu = _m2ll(co, z, r["nugget"], r["psill"], r["range"], reml)
    assert r["neg2loglik"] == pytest.approx(v, rel=1e-7)
    assert float(r["beta"][0]) == pytest.approx(mu, abs=1e-7)
    base = [r["nugget"], r["psill"], r["range"]]
    for i in range(3):
        for sgn in (1, -1):
            p = base[:]
            p[i] *= math.exp(sgn * 1e-3)
            assert _m2ll(co, z, *p, reml)[0] >= v - 1e-7
    assert r["sill"] == pytest.approx(r["nugget"] + r["psill"], rel=1e-15)
    assert r["comparable_across_mean_models"] is (method == "ml")


def test_spml_edge():
    """Unknown model or method, and a constant field, raise."""
    co, z = _data()
    with pytest.raises(ValueError):
        schabenberger_ml_variogram(co, z, "matern")
    with pytest.raises(ValueError):
        schabenberger_ml_variogram(co, z, method="mom")
    with pytest.raises(ValueError):
        schabenberger_ml_variogram(co, [1.0] * len(z))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.spml as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
