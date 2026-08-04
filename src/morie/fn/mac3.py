# morie.fn -- k02 batch (rootcoder007/morie)
"""Meta-regression on centred moderators.

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 20.  Subtracting the (weighted) mean
from each moderator leaves the slopes unchanged but makes the intercept the
predicted effect at the average study rather than at the meaningless
moderator value zero, and removes the slope/intercept correlation that
otherwise makes the intercept's standard error uninterpretable.  Fitted by
weighted least squares with weights 1/(v_i + tau^2) and the moment estimate
of the residual tau^2.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl

from ._richresult import RichResult

__all__ = ["ma_centered_predictors"]


def ma_centered_predictors(yi, vi, mods, weighted=True):
    """Meta-regression with moderators centred at their mean.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.
    mods : array-like
        Moderator matrix or vector.
    weighted : bool, default True
        Centre at the inverse-variance weighted mean rather than the plain
        mean.

    Returns
    -------
    RichResult
        estimate (intercept = effect at the average moderator), coefficients,
        se, centers, tau2_resid, QE, centered, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    m = np.atleast_2d(np.asarray(mods, dtype=float))
    if m.shape[0] != k:
        m = m.T
    w0 = 1.0 / v
    if weighted:
        ctr = [float(np.sum(w0 * m[:, j]) / np.sum(w0)) for j in range(m.shape[1])]
    else:
        ctr = [float(np.mean(m[:, j])) for j in range(m.shape[1])]
    mc = np.column_stack([m[:, j] - ctr[j] for j in range(m.shape[1])])
    x = np.column_stack([np.ones(k)] + [mc[:, j] for j in range(mc.shape[1])])
    p = x.shape[1]
    xtw = x.T * w0
    xtwxi = np.linalg.inv(np.dot(xtw, x))
    beta = np.dot(xtwxi, np.dot(xtw, y))
    resid = y - np.dot(x, beta)
    qe = float(np.sum(w0 * resid * resid))
    trp = float(np.sum(w0)) - float(np.trace(np.dot(xtwxi, np.dot(x.T * (w0 * w0), x))))
    tau2r = max(0.0, (qe - (k - p)) / trp) if trp > 0.0 else 0.0
    ws = 1.0 / (v + tau2r)
    xtws = x.T * ws
    vb = np.linalg.inv(np.dot(xtws, x))
    betar = np.dot(vb, np.dot(xtws, y))
    return RichResult(
        payload={
            "estimate": float(betar[0]),
            "coefficients": betar.tolist(),
            "se": np.sqrt(np.diag(vb)).tolist(),
            "centers": ctr,
            "tau2_resid": float(tau2r),
            "tau2_total": float(k02dl(y, v)[0]),
            "QE": qe,
            "centered": mc.tolist(),
            "n": int(k),
            "method": "Meta-regression on centred moderators (Borenstein et al. 2009, ch. 20)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_centered_predictors(y, v, [1, 2, 3, 4, 5, 6])
# >>> # centring moves the intercept but never the slope
# >>> from .marpct import ma_percent_heterogeneity_R2 as _u
# >>> assert abs(r["coefficients"][1] - _u(y, v, [1, 2, 3, 4, 5, 6])["coefficients"][1]) < 1e-12
# >>> assert abs(r["QE"] - 5.88183457856643) < 1e-10


def cheatsheet():
    return "mac3(yi, vi, mods): meta-regression on centred moderators."


macenteredpredictors = ma_centered_predictors
