# morie.fn -- k02 batch (rootcoder007/morie)
"""Baujat plot coordinates.

Source consulted: Baujat, B., Mahe, C., Pignon, J.-P. and Hill, C. (2002), A
graphical method for exploring heterogeneity in meta-analyses: application to
a meta-analysis of 65 trials, *Statistics in Medicine* 21, 2641-2652.  For
each study the plot shows

    x_i = w_i (y_i - mu_F)^2          contribution to Cochran's Q
    y_i = (mu_F - mu_F(-i))^2 / Var(mu_F(-i))    influence on the pooled effect

with fixed-effect weights w_i = 1/v_i.  Studies far right drive the
heterogeneity; studies high up move the answer.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02fe

from ._richresult import RichResult

__all__ = ["ma_baujat_plot_data"]


def ma_baujat_plot_data(yi, vi):
    """Baujat heterogeneity/influence coordinates.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.

    Returns
    -------
    RichResult
        estimate (max influence), x (Q contribution), y (influence),
        pooled, Q, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    mu, _var, _sw, q, _df = k02fe(y, v)
    w = 1.0 / v
    xs = (w * (y - mu) ** 2).tolist()
    ys = []
    for i in range(k):
        keep = [j for j in range(k) if j != i]
        yd = np.asarray([y[j] for j in keep], dtype=float)
        vd = np.asarray([v[j] for j in keep], dtype=float)
        mud, vard, _s, _q, _d = k02fe(yd, vd)
        ys.append(float((mu - mud) ** 2 / vard))
    return RichResult(
        payload={
            "estimate": float(max(ys)),
            "x": [float(t) for t in xs],
            "y": ys,
            "pooled": float(mu),
            "Q": float(q),
            "n": int(k),
            "method": "Baujat plot coordinates (Baujat, Mahe, Pignon & Hill 2002)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_baujat_plot_data(y, v)
# >>> assert abs(sum(r["x"]) - r["Q"]) < 1e-12     # the x coordinates sum to Q


def cheatsheet():
    return "mabaujat(yi, vi): Baujat heterogeneity/influence plot coordinates."


mabaujatplotdata = ma_baujat_plot_data
