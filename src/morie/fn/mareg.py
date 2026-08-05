# morie.fn -- function file (rootcoder007/morie)
"""Random-effects meta-regression on study-level moderators."""

import math

from . import _macore as ma
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ma_meta_regression"]


def ma_meta_regression(yi, vi, X):
    """Explain between-study heterogeneity with study-level covariates.

    A random-effects pool reports heterogeneity; a meta-regression asks
    where it comes from.  The residual ``tau^2`` is the part the
    moderators fail to explain, and it belongs in the weights, otherwise
    the standard errors are those of a fixed-effect fit and are too small.
    The moderators are study-level, so this is ecological regression: a
    covariate that predicts the effect across studies says nothing about
    the same covariate within a study.

    Formula: ``y_i = x_i' beta + u_i + e_i``, ``Var(u) = tau^2``,
    ``Var(e_i) = v_i``, weights ``1/(v_i + tau^2)``.  ``tau^2`` is the
    moment estimator ``max(0, (Q_E - (n - p)) / (tr W - tr((X'WX)^{-1}
    X'W^2 X)))`` -- van Houwelingen, Arends & Stijnen (2002) Section 4;
    the moment form is DerSimonian & Laird's, extended to covariates.

    Parameters
    ----------
    yi : array-like, shape (n,)
        Study effect estimates.
    vi : array-like, shape (n,)
        Their sampling variances, strictly positive.
    X : array-like, shape (n, p)
        Moderator matrix; supply the intercept column yourself.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``tau2``, ``R2``, ``ll``, ``QE``, ``QM``,
        ``n``, ``p``.

    References
    ----------
    van Houwelingen, H. C., Arends, L. R. and Stijnen, T. (2002).
    Advanced methods in meta-analysis: multivariate approach and
    meta-regression.  Statistics in Medicine 21(4):589-624.
    doi:10.1002/sim.1040.
    """
    y = [float(t) for t in core.vec(yi)]
    v = [float(t) for t in core.vec(vi)]
    Xm = core.mat(X)
    n = len(y)
    if n == 0:
        raise ValueError("no studies")
    if len(v) != n or len(Xm) != n:
        raise ValueError("yi, vi and X must have the same number of rows")
    if any(t <= 0.0 for t in v):
        raise ValueError("sampling variances must be strictly positive")
    p = len(Xm[0])
    if p > n:
        raise ValueError("more moderators than studies")
    w0 = [1.0 / t for t in v]
    b0, _, _ = ma.wls(Xm, y, w0)
    resid = [y[i] - sum(Xm[i][r] * b0[r] for r in range(p)) for i in range(n)]
    QE = sum(w0[i] * resid[i] * resid[i] for i in range(n))
    XtWX = [[sum(w0[i] * Xm[i][r] * Xm[i][s] for i in range(n))
             for s in range(p)] for r in range(p)]
    XtW2X = [[sum(w0[i] * w0[i] * Xm[i][r] * Xm[i][s] for i in range(n))
              for s in range(p)] for r in range(p)]
    inv = []
    for j in range(p):
        e = [1.0 if r == j else 0.0 for r in range(p)]
        inv.append(core.ridgesolve(XtWX, e, 1e-12))
    trterm = sum(sum(inv[j][r] * XtW2X[r][j] for r in range(p))
                 for j in range(p))
    denom = sum(w0) - trterm
    tau2 = 0.0
    if denom > 0.0:
        tau2 = (QE - (n - p)) / denom
        if tau2 < 0.0:
            tau2 = 0.0
    w = [1.0 / (v[i] + tau2) for i in range(n)]
    beta, cov, _ = ma.wls(Xm, y, w)
    se = [math.sqrt(cov[j][j]) if cov[j][j] > 0.0 else float("nan")
          for j in range(p)]
    fit = [sum(Xm[i][r] * beta[r] for r in range(p)) for i in range(n)]
    ll = -0.5 * sum(math.log(2.0 * math.pi * (v[i] + tau2))
                    + (y[i] - fit[i]) ** 2 / (v[i] + tau2) for i in range(n))
    # null model: intercept only, same moment estimator
    one = [[1.0] for _ in range(n)]
    b1, _, _ = ma.wls(one, y, w0)
    r1 = [y[i] - b1[0] for i in range(n)]
    Q1 = sum(w0[i] * r1[i] * r1[i] for i in range(n))
    d1 = sum(w0) - sum(t * t for t in w0) / sum(w0)
    tau2_null = 0.0
    if d1 > 0.0:
        tau2_null = (Q1 - (n - 1)) / d1
        if tau2_null < 0.0:
            tau2_null = 0.0
    R2 = 0.0
    if tau2_null > 0.0:
        R2 = 1.0 - tau2 / tau2_null
        if R2 < 0.0:
            R2 = 0.0
    QM = Q1 - QE
    return RichResult(payload={
        "beta": beta, "se": se, "tau2": tau2, "R2": R2, "ll": ll,
        "QE": QE, "QM": QM, "tau2_null": tau2_null, "n": n, "p": p,
        "method": "Random-effects meta-regression"})


def cheatsheet():
    return "mareg: random-effects meta-regression on study-level moderators"
