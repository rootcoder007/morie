# morie.fn -- k02 batch (rootcoder007/morie)
"""Percentage of between-study heterogeneity explained by moderators (R^2).

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 20 ("Meta-Regression"); the estimator
is Raudenbush's proportion of explained variance,

    R^2 = 100 * max(0, (tau2_total - tau2_residual) / tau2_total)

``tau2_total`` is the DerSimonian-Laird value from the intercept-only model
and ``tau2_residual`` the same moment estimator applied to the weighted
meta-regression, i.e. with W = diag(1/v), P = W - W X (X'WX)^-1 X' W,

    QE = y' P y,   tau2_res = max(0, (QE - (k - p)) / trace(P))

Verified against ``metafor::rma(mods = ...)``: tau2_res 0.0150357196157075
and QE 5.88183457856643 on the fixture below.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl

from ._richresult import RichResult

__all__ = ["ma_percent_heterogeneity_R2"]


def ma_percent_heterogeneity_R2(yi, vi, mods):
    """Meta-regression R^2 with the moment estimate of residual tau^2.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.
    mods : array-like
        Moderator matrix (k by q) or a single moderator vector; an intercept
        column is added.

    Returns
    -------
    RichResult
        estimate (R^2 in percent), tau2_total, tau2_resid, QE, df_resid,
        coefficients, se, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    m = np.atleast_2d(np.asarray(mods, dtype=float))
    if m.shape[0] != k:
        m = m.T
    x = np.column_stack([np.ones(k)] + [m[:, j] for j in range(m.shape[1])])
    p = x.shape[1]
    w = 1.0 / v
    xtw = x.T * w
    xtwxi = np.linalg.inv(np.dot(xtw, x))
    beta = np.dot(xtwxi, np.dot(xtw, y))
    resid = y - np.dot(x, beta)
    qe = float(np.sum(w * resid * resid))
    trp = float(np.sum(w)) - float(np.trace(np.dot(xtwxi, np.dot(x.T * (w * w), x))))
    tau2r = max(0.0, (qe - (k - p)) / trp) if trp > 0.0 else 0.0
    tau2t = k02dl(y, v)[0]
    ws = 1.0 / (v + tau2r)
    xtws = x.T * ws
    vb = np.linalg.inv(np.dot(xtws, x))
    betar = np.dot(vb, np.dot(xtws, y))
    return RichResult(
        payload={
            "estimate": float(100.0 * max(0.0, (tau2t - tau2r) / tau2t)) if tau2t > 0.0 else 0.0,
            "tau2_total": float(tau2t),
            "tau2_resid": float(tau2r),
            "QE": qe,
            "df_resid": int(k - p),
            "coefficients": betar.tolist(),
            "se": np.sqrt(np.diag(vb)).tolist(),
            "n": int(k),
            "method": "Meta-regression R^2, proportion of tau^2 explained (Borenstein et al. 2009, ch. 20)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_percent_heterogeneity_R2(y, v, [1, 2, 3, 4, 5, 6])
# >>> assert abs(r["QE"] - 5.88183457856643) < 1e-10          # metafor rma mods
# >>> assert abs(r["tau2_resid"] - 0.0150357196157075) < 1e-12
# >>> assert r["estimate"] == 0.0    # residual tau^2 exceeds the total here


def cheatsheet():
    return "marpct(yi, vi, mods): meta-regression R^2 (percent of tau^2 explained)."


mapercentheterogeneityr2 = ma_percent_heterogeneity_R2
