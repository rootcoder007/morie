# morie.fn -- k02 batch (rootcoder007/morie)
"""Cumulative meta-analysis.

Source consulted: Lau, J., Schmid, C.H. and Chalmers, T.C. (1995), Cumulative
meta-analysis of clinical trials builds evidence for exemplary medical care,
*Journal of Clinical Epidemiology* 48(1), 45-57, and the original application
in Lau et al. (1992).  The studies are sorted by ``order`` (publication year
by default) and the random-effects summary is recomputed after each addition,
so row j is the meta-analysis that would have been available once study j had
appeared.  Each step uses DerSimonian-Laird.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02z

from ._richresult import RichResult

__all__ = ["ma_cumulative"]


def ma_cumulative(yi, vi, order=None, level=0.95):
    """Cumulative random-effects meta-analysis.

    Parameters
    ----------
    yi, vi : array-like
        Study effects and their within-study variances.
    order : array-like, optional
        Sort key (e.g. year).  Input order if omitted.
    level : float, default 0.95
        Confidence level for each cumulative interval.

    Returns
    -------
    RichResult
        estimate (final pooled value), cumulative, se, ci_lower, ci_upper,
        tau2, order_index, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    if order is None:
        idx = list(range(k))
    else:
        idx = np.argsort(np.asarray(order, dtype=float)).tolist()
    crit = k02z(0.5 + 0.5 * float(level))
    est = []
    ses = []
    lo = []
    hi = []
    t2s = []
    for j in range(1, k + 1):
        sub = idx[:j]
        ys = np.asarray([y[i] for i in sub], dtype=float)
        vs = np.asarray([v[i] for i in sub], dtype=float)
        if j == 1:
            t2 = 0.0
            mu = float(ys[0])
            var = float(vs[0])
        else:
            t2, mu, var, _q, _df = k02dl(ys, vs)
        se = float(np.sqrt(var))
        est.append(float(mu))
        ses.append(se)
        lo.append(float(mu - crit * se))
        hi.append(float(mu + crit * se))
        t2s.append(float(t2))
    return RichResult(
        payload={
            "estimate": est[-1],
            "cumulative": est,
            "se": ses,
            "ci_lower": lo,
            "ci_upper": hi,
            "tau2": t2s,
            "order_index": [int(i) for i in idx],
            "n": int(k),
            "method": "Cumulative random-effects meta-analysis (Lau, Schmid & Chalmers 1995)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_cumulative(y, v)
# >>> assert abs(r["cumulative"][0] - 0.10) < 1e-15      # first step is study 1
# >>> assert abs(r["estimate"] - 0.0920094772579361) < 1e-13   # last = full DL


def cheatsheet():
    return "macum(yi, vi, order): cumulative random-effects meta-analysis."


macumulative = ma_cumulative
