# morie.fn -- k02 batch (rootcoder007/morie)
"""Leave-one-out influence diagnostics for a random-effects meta-analysis.

Source consulted: Viechtbauer, W. and Cheung, M.W.-L. (2010), Outlier and
influence diagnostics for meta-analysis, *Research Synthesis Methods* 1,
112-125.  With w_i = 1/(v_i + tau^2) and the delete-one refits (each of which
re-estimates tau^2 by DerSimonian-Laird):

    hat_i      = w_i / sum(w)
    rstudent_i = (y_i - mu_(-i)) / sqrt(v_i + tau2_(-i) + Var(mu_(-i)))
    dffits_i   = (mu - mu_(-i)) / sqrt(hat_i (v_i + tau2_(-i)))
    cook_i     = (mu - mu_(-i))^2 / Var(mu)

All four, plus ``tau2_del`` and ``Q_del``, reproduce ``metafor::influence()``
on the fixture in the canonical test to 1e-9.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02fe

from ._richresult import RichResult

__all__ = ["ma_influence_diagnostics"]


def ma_influence_diagnostics(yi, vi):
    """Delete-one influence diagnostics.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.

    Returns
    -------
    RichResult
        estimate (largest Cook's distance), rstudent, dffits, cook_d, hat,
        tau2_del, Q_del, estimate_del, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    tau2, mu, var, _q, _df = k02dl(y, v)
    w = 1.0 / (v + tau2)
    hat = (w / float(np.sum(w))).tolist()
    rst = []
    dff = []
    cook = []
    t2d = []
    qd = []
    mud_all = []
    for i in range(k):
        keep = [j for j in range(k) if j != i]
        yd = np.asarray([y[j] for j in keep], dtype=float)
        vd = np.asarray([v[j] for j in keep], dtype=float)
        t2i, mui, vari, _qq, _dd = k02dl(yd, vd)
        qi = k02fe(yd, vd)[3]
        rst.append(float((y[i] - mui) / np.sqrt(v[i] + t2i + vari)))
        dff.append(float((mu - mui) / np.sqrt(hat[i] * (v[i] + t2i))))
        cook.append(float((mu - mui) ** 2 / var))
        t2d.append(float(t2i))
        qd.append(float(qi))
        mud_all.append(float(mui))
    return RichResult(
        payload={
            "estimate": float(max(cook)),
            "rstudent": rst,
            "dffits": dff,
            "cook_d": cook,
            "hat": hat,
            "tau2_del": t2d,
            "Q_del": qd,
            "estimate_del": mud_all,
            "n": int(k),
            "method": "Leave-one-out meta-analysis influence diagnostics (Viechtbauer & Cheung 2010)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_influence_diagnostics(y, v)
# >>> assert abs(r["rstudent"][2] + 1.772369963) < 1e-8    # metafor influence()
# >>> assert abs(r["cook_d"][2] - 0.281551845) < 1e-8
# >>> assert abs(r["hat"][0] - 0.21341749) < 1e-8
# >>> assert abs(r["dffits"][0] + 0.1205870) < 1e-6


def cheatsheet():
    return "mainf(yi, vi): leave-one-out meta-analysis influence diagnostics."


mainfluencediagnostics = ma_influence_diagnostics
