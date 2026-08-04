# morie.fn -- k02 batch shared helpers (rootcoder007/morie)
"""Internal helpers shared by the k02 batch.  Not part of the public API.

Mirrors ``r-package/morie/R/k02util.R`` statement for statement so the three
arms (Python, morie R, rmorie) agree to machine precision.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as _st

__all__ = []


def k02fe(y, v):
    """Inverse-variance fixed-effect summary.

    Returns ``(mu, var, sumw, Q, df)`` where ``mu = sum(w y)/sum(w)``,
    ``w = 1/v``, ``var = 1/sum(w)`` and ``Q = sum(w (y - mu)^2)`` is
    Cochran's homogeneity statistic on ``df = k - 1`` degrees of freedom.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = 1.0 / v
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    q = float(np.sum(w * (y - mu) ** 2))
    return mu, 1.0 / sw, sw, q, len(y) - 1


def k02dl(y, v):
    """DerSimonian-Laird moment estimator of tau^2 and the RE summary.

    Returns ``(tau2, mu, var, Q, df)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    mu, _var, sw, q, df = k02fe(y, v)
    w = 1.0 / v
    c = sw - float(np.sum(w * w)) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0.0 else 0.0
    ws = 1.0 / (v + tau2)
    sws = float(np.sum(ws))
    mur = float(np.sum(ws * y)) / sws
    return tau2, mur, 1.0 / sws, q, df


def k02mm(y, v, tau0):
    """Generalised method-of-moments tau^2 at working weights 1/(v + tau0).

    DerSimonian and Kacker (2007) equation (6): with a_i = 1/(v_i + tau0),

        tau2 = [ sum a_i (y_i - ybar_a)^2 - sum a_i v_i + sum a_i^2 v_i / sum a_i ]
               / [ sum a_i - sum a_i^2 / sum a_i ]

    which collapses exactly to DerSimonian-Laird when tau0 = 0.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    a = 1.0 / (v + tau0)
    sa = float(np.sum(a))
    sa2 = float(np.sum(a * a))
    yb = float(np.sum(a * y)) / sa
    num = float(np.sum(a * (y - yb) ** 2)) - float(np.sum(a * v)) + float(np.sum(a * a * v)) / sa
    den = sa - sa2 / sa
    return max(0.0, num / den) if den > 0.0 else 0.0


def k02z(p):
    return float(_st.norm.ppf(p))


def k02tq(p, df):
    return float(_st.t.ppf(p, df))


def k02p2z(z):
    return 2.0 * float(_st.norm.sf(abs(z)))


def k02p2t(tv, df):
    return 2.0 * float(_st.t.sf(abs(tv), df))


def k02pchi(q, df):
    return float(_st.chi2.sf(q, df))
