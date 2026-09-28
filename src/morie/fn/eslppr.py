# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Projection pursuit regression (ESL Sec 11.2, eqs 11.1-11.4)."""

import math

from ._richresult import RichResult
from .eslgam import _spline_smooth
from .linsys import _householder_ls

__all__ = ["esl_projection_pursuit"]


def _ridge_eval(term, t):
    """Value and slope of a ridge function (natural cubic spline, linear beyond the end knots) at t."""
    u, f, g = term["knots"], term["values"], term["gamma"]
    m = len(u)
    if m == 1:
        return f[0], 0.0
    if t <= u[0] or t >= u[-1]:
        k = 0 if t <= u[0] else m - 2
        h = u[k + 1] - u[k]
        if t <= u[0]:
            d = (f[1] - f[0]) / h - h * (2 * g[0] + g[1]) / 6
            return f[0] + d * (t - u[0]), d
        d = (f[-1] - f[-2]) / h + h * (g[-2] + 2 * g[-1]) / 6
        return f[-1] + d * (t - u[-1]), d
    lo, hi = 0, m - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u[mid] <= t:
            lo = mid
        else:
            hi = mid
    k = lo
    h = u[k + 1] - u[k]
    a, b = t - u[k], u[k + 1] - t
    val = (a * f[k + 1] + b * f[k]) / h - a * b / 6 * ((1 + a / h) * g[k + 1] + (1 + b / h) * g[k])
    der = (f[k + 1] - f[k]) / h + ((3 * a * a - h * h) * g[k + 1] - (3 * b * b - h * h) * g[k]) / (6 * h)
    return val, der


def _fit_ridge(X, r, omega, penalty):
    v = [sum(a * b for a, b in zip(x, omega)) for x in X]
    fit, u, f, g = _spline_smooth(v, r, penalty, full=True)
    return {"omega": omega, "knots": u, "values": f, "gamma": g}, fit


def esl_projection_pursuit(X, y, M=2, penalty=1.0, max_iter=50, tol=1e-8, backfit=10, newdata=None):
    r"""Projection pursuit regression (Friedman & Tukey; ESL Sec 11.2).

    Fits :math:`f(X) = \bar y + \sum_{m=1}^M g_m(\omega_m^T X)` (11.1) forward
    stage-wise, minimising :math:`\sum_i [y_i - \sum_m g_m(\omega_m^T x_i)]^2` (11.2).
    For each new term, alternate (a) a cubic smoothing spline (``penalty``) of
    the current residual on :math:`v_i = \omega^T x_i`, and (b) the Gauss-Newton
    update (11.3)-(11.4): weighted least squares of
    :math:`\omega_{old}^T x_i + (r_i - g(\omega_{old}^T x_i))/g'(\omega_{old}^T x_i)` on
    :math:`x_i`, weights :math:`g'^2`, no intercept -- solved as the equivalent
    unweighted regression of :math:`g' \omega_{old}^T x_i + r_i - g` on
    :math:`g' x_i`, so a zero slope never divides. The step is halved (up to
    ten times) if it raises the residual sum of squares, and :math:`\omega` is
    rescaled to unit length. After each new term the ridge functions are
    readjusted by up to ``backfit`` backfitting passes with the directions
    fixed (ESL's implementation notes); :math:`\omega_m` is not readjusted.

    Parameters
    ----------
    X : N x p nested sequence
    y : sequence of N responses
    M : int
        Number of ridge terms.
    penalty : float
        Smoothing-spline penalty (> 0) on the scale of :math:`\omega^T x`.
    max_iter, tol
        Gauss-Newton controls per term (stop when :math:`1-|\omega_{new}^T\omega_{old}| <` ``tol``).
    backfit : int
        Maximum backfitting passes after each term (0 disables).
    newdata : nested sequence, optional
        Points at which to evaluate the fitted model.

    Returns
    -------
    RichResult
        ``omega`` (M unit directions), ``fitted``, ``residuals``, ``rss``,
        ``rss_path`` (RSS after each term), ``intercept``, ``terms`` (knots,
        values and second derivatives of each ridge function), ``predicted``.

    References
    ----------
    Friedman, J. H. & Tukey, J. W. (1974). IEEE Trans. Computers C-23, 881-890.
    Hastie, T., Tibshirani, R. & Friedman, J. (2009). The Elements of
    Statistical Learning, 2nd ed., Sec 11.2.

    Examples
    --------
    >>> X = [[i / 5 - 2, ((7 * i) % 11) / 5 - 1] for i in range(21)]
    >>> y = [(a + b) ** 2 for a, b in X]
    >>> r = esl_projection_pursuit(X, y, M=1, penalty=0.01)
    >>> round(abs(r["omega"][0][0] - r["omega"][0][1]), 2)
    0.0
    """
    X = [[float(v) for v in x] for x in X]
    y = [float(v) for v in y]
    n, p = len(X), len(X[0]) if X else 0
    if n < 4 or p < 1 or len(y) != n or M < 1 or penalty <= 0:
        raise ValueError("need N >= 4 matching rows, M >= 1 and penalty > 0")
    ybar = sum(y) / n
    r = [v - ybar for v in y]
    terms, fits, path = [], [], []
    xbar = [sum(c) / n for c in zip(*X)]
    Xc = [[a - b for a, b in zip(x, xbar)] for x in X]
    for _ in range(M):
        omega, _ = _householder_ls(Xc, r)  # start from the least-squares direction of the residual
        nrm = math.sqrt(sum(w * w for w in omega))
        omega = [w / nrm for w in omega] if nrm > 0 else [1.0] + [0.0] * (p - 1)
        term, fit = _fit_ridge(X, r, omega, penalty)
        rss = sum((a - b) ** 2 for a, b in zip(r, fit))
        for _ in range(max_iter):
            v = [sum(a * b for a, b in zip(x, omega)) for x in X]
            gd = [_ridge_eval(term, t) for t in v]
            rows = [[d * xx for xx in x] for (_, d), x in zip(gd, X)]
            z = [d * vi + ri - gi for (gi, d), vi, ri in zip(gd, v, r)]
            new, _ = _householder_ls(rows, z)
            step = 1.0
            for _ in range(11):
                cand = [o + step * (w - o) for o, w in zip(omega, new)]
                nc = math.sqrt(sum(w * w for w in cand))
                cand = [w / nc for w in cand]
                cterm, cfit = _fit_ridge(X, r, cand, penalty)
                crss = sum((a - b) ** 2 for a, b in zip(r, cfit))
                if crss <= rss:
                    break
                step /= 2
            else:
                break
            moved = 1 - abs(sum(a * b for a, b in zip(cand, omega)))
            omega, term, fit, rss = cand, cterm, cfit, crss
            if moved < tol:
                break
        terms.append(term)
        fits.append(fit)
        r = [a - b for a, b in zip(r, fit)]
        for _ in range(backfit if len(terms) > 1 else 0):
            change = 0.0
            for m, tm in enumerate(terms):
                part = [a + b for a, b in zip(r, fits[m])]
                terms[m], nf = _fit_ridge(X, part, tm["omega"], penalty)
                change = max(change, max(abs(a - b) for a, b in zip(nf, fits[m])))
                fits[m] = nf
                r = [a - b for a, b in zip(part, nf)]
            if change < 1e-10:
                break
        path.append(sum(v * v for v in r))
    fitted = [a - b for a, b in zip(y, r)]
    pred = None
    if newdata is not None:
        pred = [
            ybar + sum(_ridge_eval(t, sum(a * b for a, b in zip(x, t["omega"])))[0] for t in terms) for x in newdata
        ]
    return RichResult(
        title="Projection pursuit regression",
        summary_lines=[("terms", M), ("rss", path[-1])],
        payload={
            "omega": [t["omega"] for t in terms],
            "fitted": fitted,
            "residuals": r,
            "rss": path[-1],
            "rss_path": path,
            "intercept": ybar,
            "terms": terms,
            "predicted": pred,
        },
    )


def cheatsheet() -> str:
    return "esl_projection_pursuit -> projection pursuit regression with smoothing-spline ridge functions (ESL 11.1-11.4)."
