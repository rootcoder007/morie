"""Regression quantiles (Koenker & Bassett 1978)."""

import itertools

from ._richresult import RichResult

__all__ = ["quanrg", "quantile_regression"]


def _solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c] / m[c][c]
                for j in range(c, n + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def _check_loss(res, theta):
    # the defining objective of Koenker & Bassett (1978), Sec. 2:
    # theta |u| on non-negative residuals, (1 - theta)|u| on negatives
    return sum(theta * r if r >= 0 else (theta - 1.0) * r for r in res)


def quanrg(y, X=None, theta=0.5):
    """
    Exact regression quantile estimate.

    Koenker & Bassett (1978): the theta-th regression quantile
    minimizes sum_t rho_theta(y_t - x_t' b) with the asymmetric
    absolute loss (their Sec. 2 minimization problem: weight theta
    on non-negative residuals, 1 - theta on negative ones); theta =
    1/2 gives the least-absolute-error (LAD) estimator.  Their
    Theorem 3.1: when X has rank K, the solution set contains an
    element of the form b* = X(h)^{-1} y(h) for some K-element
    subset h -- i.e., an exact fit through K observations -- and is
    the convex hull of such basic solutions.  This implementation is
    therefore EXACT: it enumerates all K-subsets with nonsingular
    X(h), evaluates the objective at each basic solution, and
    returns the minimizer (suitable for K <= 3 and moderate n; cost
    C(n, K) * n).

    Sources
    -------
    Koenker, R. & Bassett, G. (1978). Regression quantiles.
    *Econometrica*, 46(1), 33-50, Sec. 2 (defining minimization)
    and Theorem 3.1 (basic solutions) (local copy
    fetched-wave3/Koenker-RegressionQuantiles-1978.pdf).
    Koenker, R. (2005). *Quantile Regression*. Cambridge University
    Press (delivered; the field's standard monograph).

    Parameters
    ----------
    y : sequence of float
        Responses.
    X : sequence of rows, optional
        Regressors WITHOUT intercept (added automatically); default
        location model (intercept only) whose solution is the
        ordinary sample quantile.
    theta : float
        Quantile level in (0, 1).

    Returns
    -------
    RichResult
        Keys: coefficients, objective, basis (row indices of the
        exact-fit observations), n_bases_checked, theta.
    """
    yv = [float(v) for v in y]
    n = len(yv)
    if X is None:
        rows = [[1.0] for _ in range(n)]
    else:
        rows = [[1.0] + [float(v) for v in r] for r in X]
        if len(rows) != n:
            raise ValueError("X must have one row per observation")
    k = len(rows[0])
    theta = float(theta)
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0, 1)")
    if k > 3:
        raise ValueError("basis enumeration supports at most 3 "
                         "coefficients (intercept + 2 regressors)")
    if n < k + 1:
        raise ValueError("need more observations than coefficients")
    best = None
    checked = 0
    for h in itertools.combinations(range(n), k):
        a = [rows[i] for i in h]
        b = [yv[i] for i in h]
        beta = _solve(a, b)
        if beta is None:
            continue
        checked += 1
        res = [yv[i] - sum(bb * v for bb, v in zip(beta, rows[i]))
               for i in range(n)]
        obj = _check_loss(res, theta)
        if best is None or obj < best[0] - 1e-15:
            best = (obj, beta, h)
    if best is None:
        raise ValueError("design matrix has no nonsingular basis")
    obj, beta, h = best
    return RichResult(payload={
        "coefficients": beta,
        "objective": obj,
        "basis": list(h),
        "n_bases_checked": checked,
        "theta": theta,
        "method": "exact regression quantile (K&B 1978 Thm 3.1 bases)",
    })


# long descriptive alias (stub-era name)
quantile_regression = quanrg


def cheatsheet():
    return "quanrg: enumerate X(h)^-1 y(h) bases, min sum rho_theta(residual)"
