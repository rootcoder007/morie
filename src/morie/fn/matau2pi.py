# morie.fn -- k02 batch (rootcoder007/morie)
"""Prediction interval for a new study from tau^2.

Source consulted: Higgins, J.P.T., Thompson, S.G. and Spiegelhalter, D.J.
(2009), A re-evaluation of random-effects meta-analysis, *JRSS Series A*
172(1), 137-159, section 3.1.  The interval for the effect in a new study is

    mu +/- t_{k-2} sqrt(tau^2 + SE(mu)^2)

The k - 2 degrees of freedom are the paper's own recommendation (one lost for
mu, one for tau^2).  ``metafor::predict`` uses k - 1 with the Knapp-Hartung
standard error; that variant is returned as ``pi_lower_km1``/``pi_upper_km1``
so the two conventions can be compared rather than confused.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02tq

from ._richresult import RichResult

__all__ = ["ma_tau2_predict_interval"]


def ma_tau2_predict_interval(yi, vi, level=0.95):
    """Higgins-Thompson-Spiegelhalter prediction interval.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.
    level : float, default 0.95
        Interval level.

    Returns
    -------
    RichResult
        estimate (mu), se, tau2, spread, pi_lower, pi_upper, pi_lower_km1,
        pi_upper_km1, df, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    tau2, mu, var, _q, _df = k02dl(y, v)
    se = float(np.sqrt(var))
    spread = float(np.sqrt(tau2 + var))
    c2 = k02tq(0.5 + 0.5 * float(level), k - 2)
    c1 = k02tq(0.5 + 0.5 * float(level), k - 1)
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "tau2": float(tau2),
            "spread": spread,
            "pi_lower": float(mu - c2 * spread),
            "pi_upper": float(mu + c2 * spread),
            "pi_lower_km1": float(mu - c1 * spread),
            "pi_upper_km1": float(mu + c1 * spread),
            "df": int(k - 2),
            "n": int(k),
            "method": "Prediction interval for a new study (Higgins, Thompson & Spiegelhalter 2009, sec. 3.1)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_tau2_predict_interval(y, v)
# >>> assert r["pi_lower"] < r["estimate"] < r["pi_upper"]
# >>> assert r["pi_upper"] > r["pi_upper_km1"]   # k-2 is wider than k-1


def cheatsheet():
    return "matau2pi(yi, vi): prediction interval for a new study."


matau2predictinterval = ma_tau2_predict_interval
