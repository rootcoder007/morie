# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Pairs bootstrap confidence intervals for quantile regression.

Koenker, R. (2005), *Quantile Regression*, Econometric Society
Monograph 38, Cambridge University Press.  Section 3.9 covers
resampling for quantile regression; the (x, y)-pair bootstrap is the
default there because the asymptotic covariance of the quantile
regression estimator involves the conditional density of the response at
the quantile, a nuisance nobody wants to estimate, and pair resampling
sidesteps it entirely while remaining valid under a stochastic design
and heteroskedasticity.

The fit minimises Koenker and Bassett's check function

    sum_i rho_tau( y_i - x_i' beta ),   rho_tau(u) = u (tau - 1{u < 0}),

by iteratively reweighted least squares with the Schlossmacher weights
w_i = tau / max(r_i, eps) for r_i > 0 and (1 - tau) / max(-r_i, eps)
otherwise, which is the standard smooth surrogate for the linear
program.  The eps floor is what keeps a residual that lands exactly on
the fitted plane from producing an infinite weight; it makes the
solution an approximation to the LP vertex, accurate to well beyond the
resampling noise but not exact, and that is stated here rather than
hidden.

Anchors: with an intercept-only design the check-function minimiser is
the tau-th sample quantile of y, computed here by base R / by direct
sorting rather than through the fitter; and the fitted objective must
not exceed the objective at the OLS coefficients, since the OLS point is
feasible for the same minimisation.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_quantile_regression"]


def qrfit(Xm, yy, tau, maxit=200, tol=1e-10, eps=1e-6):
    """Quantile regression coefficients by Schlossmacher IRLS."""
    n = len(yy)
    p = len(Xm[0])
    b = core.lstsq(Xm, yy)
    for _ in range(int(maxit)):
        w = []
        for i in range(n):
            r = yy[i] - sum(Xm[i][j] * b[j] for j in range(p))
            if r > 0.0:
                w.append(tau / max(r, eps))
            else:
                w.append((1.0 - tau) / max(-r, eps))
        Xs = [[math.sqrt(w[i]) * Xm[i][j] for j in range(p)] for i in range(n)]
        ys = [math.sqrt(w[i]) * yy[i] for i in range(n)]
        nb = core.lstsq(Xs, ys)
        d = max(abs(nb[j] - b[j]) for j in range(p))
        b = nb
        if d < tol:
            break
    return b


def check_loss(Xm, yy, b, tau):
    """sum_i rho_tau(y_i - x_i'b)."""
    n = len(yy)
    p = len(b)
    s = 0.0
    for i in range(n):
        u = yy[i] - sum(Xm[i][j] * b[j] for j in range(p))
        s += u * (tau - (1.0 if u < 0.0 else 0.0))
    return s


def boot_quantile_regression(X, y, tau=0.5, B=200, alpha=0.05, seed=1):
    """Pairs bootstrap for the quantile regression coefficient vector.

    Parameters
    ----------
    X : array-like
        The n x p design.
    y : array-like
        The n responses.
    tau : float
        Quantile level, strictly between 0 and 1.
    B : int
        Replicates.
    alpha : float
        Two-sided error rate.
    seed : int
        Seed for the shared deterministic stream.

    Returns
    -------
    RichResult
        ``beta_b``, ``beta_hat``, ``se``, ``lo``/``hi``, ``loss``
        (check-function value at the fit), ``tau``, ``n``, ``p``, ``B``.
    """
    from . import _tail1core as C

    Xm = core.mat(X)
    yy = core.vec(y)
    n = core.nrow(Xm)
    p = core.ncol(Xm)
    if n != len(yy):
        raise ValueError("boot_quantile_regression: X and y have different lengths")
    if n <= p:
        raise ValueError("boot_quantile_regression: need more rows than columns")
    t = float(tau)
    if not (0.0 < t < 1.0):
        raise ValueError("boot_quantile_regression: tau must lie strictly between 0 and 1")
    if int(B) < 2:
        raise ValueError("boot_quantile_regression: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_quantile_regression: alpha must lie strictly between 0 and 1")
    bh = qrfit(Xm, yy, t)
    g = C.Lcg(seed)
    reps = []
    for _ in range(int(B)):
        idx = []
        for _i in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            idx.append(j)
        reps.append(qrfit([Xm[j] for j in idx], [yy[j] for j in idx], t))
    se = []
    lo = []
    hi = []
    for j in range(p):
        col = [r[j] for r in reps]
        se.append(core.sd(col, 1))
        lo.append(core.quantile7(col, a / 2.0))
        hi.append(core.quantile7(col, 1.0 - a / 2.0))
    return RichResult(
        title="Pairs bootstrap for quantile regression",
        summary_lines=[("tau", t), ("n", n), ("p", p), ("B", int(B))],
        payload={
            "beta_b": reps,
            "beta_hat": bh,
            "se": se,
            "lo": lo,
            "hi": hi,
            "loss": check_loss(Xm, yy, bh, t),
            "tau": t,
            "n": n,
            "p": p,
            "B": int(B),
            "estimate": bh[0],
            "method": "Koenker (2005) Quantile Regression, CUP, sec. 3.9 (xy-pair bootstrap)",
        },
    )


def cheatsheet():
    return "btnpqr: pair-resample for QR; the asymptotic covariance needs a conditional density, this does not"
