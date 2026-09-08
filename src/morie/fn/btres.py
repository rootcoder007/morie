# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Residual bootstrap for OLS coefficients.

Freedman, D. A. (1981), "Bootstrapping Regression Models", *The Annals
of Statistics* 9(6), 1218-1228, doi:10.1214/aos/1176345638 (verified
against Crossref).

The design is held fixed and only the errors are resampled:

    y* = X beta_hat + r*,   r*_i drawn with replacement from the
                            centred OLS residuals,

then OLS is refit on (X, y*).  This conditions on X, which is the right
thing to do when X is genuinely fixed, but it buys that by assuming the
errors are iid -- under heteroskedasticity it is inconsistent and the
pairs or wild bootstrap is required instead.

Because the design never changes, the conditional moments are available
in closed form and are this module's anchor:

    E*[beta*]   = beta_hat,
    Var*(beta*) = sigma_tilde^2 (X'X)^{-1},
    sigma_tilde^2 = sum r_i^2 / n,

using the resampling distribution's own variance (divisor n, not n - p).
``var_closed`` reports the diagonal.  Note that this is the homoskedastic
formula with a DOWNWARD-biased scale: the residual bootstrap inherits the
OLS residuals' shrinkage, which is exactly why Freedman's rescaled
variant divides the residuals by sqrt(1 - h_ii).  ``rescale=True``
selects that variant.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_residual_regression"]


def boot_residual_regression(X, y, B=200, seed=1, alpha=0.05, rescale=False):
    """Residual-resampling bootstrap for the OLS coefficient vector.

    Parameters
    ----------
    X : array-like
        The n x p design.
    y : array-like
        The n responses.
    B : int
        Replicates.
    seed : int
        Seed for the shared deterministic stream.
    alpha : float
        Two-sided error rate.
    rescale : bool
        Divide each residual by sqrt(1 - h_ii) before resampling, which
        removes the leverage-induced shrinkage of the OLS residuals.

    Returns
    -------
    RichResult
        ``beta_b``, ``beta_hat``, ``resid``, ``se``, ``lo``/``hi``,
        ``var_closed`` (per coefficient), ``sigma2_tilde``, ``n``,
        ``p``, ``B``.
    """
    from . import _tail1core as C

    Xm = core.mat(X)
    yy = core.vec(y)
    n = core.nrow(Xm)
    p = core.ncol(Xm)
    if n != len(yy):
        raise ValueError("boot_residual_regression: X and y have different lengths")
    if n <= p:
        raise ValueError("boot_residual_regression: need more rows than columns")
    if int(B) < 2:
        raise ValueError("boot_residual_regression: need at least two replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_residual_regression: alpha must lie strictly between 0 and 1")
    bh = core.lstsq(Xm, yy)
    fit = [sum(Xm[i][j] * bh[j] for j in range(p)) for i in range(n)]
    res = [yy[i] - fit[i] for i in range(n)]
    XtXinv = _xtxinv(Xm, n, p)
    if rescale:
        h = [sum(Xm[i][j] * sum(XtXinv[j][k] * Xm[i][k] for k in range(p)) for j in range(p))
             for i in range(n)]
        res = [res[i] / math.sqrt(max(1.0 - h[i], 1e-12)) for i in range(n)]
    rb = core.mean(res)
    res = [u - rb for u in res]
    s2 = sum(u * u for u in res) / n
    g = C.Lcg(seed)
    reps = []
    for _ in range(int(B)):
        ys = []
        for i in range(n):
            j = int(g.unif() * n)
            if j >= n:
                j = n - 1
            ys.append(fit[i] + res[j])
        reps.append(core.lstsq(Xm, ys))
    se = []
    lo = []
    hi = []
    vc = []
    for j in range(p):
        col = [r[j] for r in reps]
        se.append(core.sd(col, 1))
        lo.append(core.quantile7(col, a / 2.0))
        hi.append(core.quantile7(col, 1.0 - a / 2.0))
        vc.append(s2 * XtXinv[j][j])
    return RichResult(
        title="Residual bootstrap for OLS",
        summary_lines=[("n", n), ("p", p), ("B", int(B))],
        payload={
            "beta_b": reps,
            "beta_hat": bh,
            "resid": res,
            "se": se,
            "lo": lo,
            "hi": hi,
            "var_closed": vc,
            "sigma2_tilde": s2,
            "n": n,
            "p": p,
            "B": int(B),
            "estimate": bh[0],
            "method": "Freedman (1981) Ann. Statist. 9(6):1218-1228, residual resampling",
        },
    )


def _xtxinv(Xm, n, p):
    """(X'X)^{-1} by solving the ridge-stabilised system against each unit vector."""
    A = core.crossprod(Xm)
    cols = []
    for j in range(p):
        e = [1.0 if k == j else 0.0 for k in range(p)]
        cols.append(core.ridgesolve(A, e))
    return [[cols[j][i] for j in range(p)] for i in range(p)]


def cheatsheet():
    return "btres: hold X fixed, resample residuals; Var*(beta*) = (sum r^2/n) (X'X)^-1 exactly"
