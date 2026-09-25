"""Tests for tmlavp.tmle_average_predictiveness (algorithm-agnostic VIM)."""

import math
import statistics

import pytest

from morie.fn import _array_core as np
from morie.fn._vimp import predictiveness
from morie.fn.tmlavp import tmle_average_predictiveness


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _ols(Xtr, ytr):
    """Fitting function: least squares with an intercept, as a predictor."""
    Z = [[1.0] + [float(v) for v in row] for row in Xtr.tolist()]
    t = [float(v) for v in ytr.tolist()]
    q = len(Z[0])
    b = _solve([[sum(z[r] * z[c] for z in Z) for c in range(q)] for r in range(q)],
               [sum(z[r] * y for z, y in zip(Z, t)) for r in range(q)])
    return lambda Xn: [b[0] + sum(u * v for u, v in zip(row, b[1:])) for row in np.asarray(Xn).tolist()]


def _data(n=120):
    X = [[math.sin(1.1 * k), math.cos(0.7 * k), math.sin(0.3 * k + 1)] for k in range(n)]
    y = [2 * x[0] + 0.5 * x[1] + 0.4 * math.sin(9.7 * k) for k, x in enumerate(X)]
    return y, X


def _r2(y, p):
    """R^2 = 1 - MSE/Var and its delta-method gradient."""
    n = len(y)
    mse = sum((a - b) ** 2 for a, b in zip(y, p)) / n
    mu = sum(y) / n
    var = sum((a - mu) ** 2 for a in y) / n
    g = [-(((a - b) ** 2 - mse) / var - mse * ((a - mu) ** 2 - var) / var ** 2) for a, b in zip(y, p)]
    return 1 - mse / var, g


def _reference(y, X, s, K=4, seed=3):
    """Williamson et al. (2023) Algorithm 3: disjoint halves for the full
    and reduced predictiveness, each cross-fitted over K folds; variance
    eta_full/n_a + eta_red/n_b, eta the fold mean of mean(grad^2).  The
    split and folds replay the module's seeded draws in the same order."""
    n = len(y)
    rng = np.random.default_rng(seed)
    half = [int(v) for v in rng.permutation(n).tolist()]
    keep = [j for j in range(len(X[0])) if j not in s]

    def cf(idx, cols):
        folds = [int(v) % K for v in rng.permutation(len(idx)).tolist()]
        vals, etas = [], []
        for k in range(K):
            te = [idx[i] for i in range(len(idx)) if folds[i] == k]
            tr = [idx[i] for i in range(len(idx)) if folds[i] != k]
            fit = _ols(np.asarray([[X[i][c] for c in cols] for i in tr]), np.asarray([y[i] for i in tr]))
            v, g = _r2([y[i] for i in te], fit(np.asarray([[X[i][c] for c in cols] for i in te])))
            vals.append(v)
            etas.append(sum(x * x for x in g) / len(g))
        return statistics.fmean(vals), statistics.fmean(etas)
    a, b = half[: n // 2], half[n // 2:]
    vf, ef = cf(a, list(range(len(X[0]))))
    vr, er = cf(b, keep)
    return vf - vr, math.sqrt(ef / len(a) + er / len(b)), vf, vr


def test_tmlavp_basic():
    """Estimate, standard error and both predictiveness values match the
    independent replay of the sample-split, cross-fitted R^2 contrast."""
    y, X = _data()
    psi, se, vf, vr = _reference(y, X, [0])
    r = tmle_average_predictiveness(y, [0], X, f=_ols, loss="r_squared", n_folds=4, seed=3)
    assert r["estimate"] == pytest.approx(psi, abs=1e-9)
    assert r["se"] == pytest.approx(se, rel=1e-9)
    assert r["v_full"] == pytest.approx(vf, abs=1e-9)
    assert r["v_reduced"] == pytest.approx(vr, abs=1e-9)
    assert r["p_value"] == pytest.approx(0.5 * math.erfc(psi / se / math.sqrt(2)), rel=1e-9)
    assert r["null_inference_valid"] is True


def test_tmlavp_edge():
    """Unknown measures raise; the AUC plug-in equals the Mann-Whitney
    count with ties halved and its gradient is mean-zero."""
    y, X = _data()
    with pytest.raises(ValueError):
        tmle_average_predictiveness(y, [0], X, loss="mse")
    yb = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    pr = [0.9, 0.2, 0.4, 0.7, 0.4, 0.8, 0.6]
    wins = sum((a > b) + 0.5 * (a == b) for a, t in zip(pr, yb) if t for b, u in zip(pr, yb) if not u)
    v, g = predictiveness(yb, pr, "auc")
    assert v == pytest.approx(wins / 12.0, abs=1e-15)
    assert float(np.mean(g)) == pytest.approx(0.0, abs=1e-12)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.tmlavp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
