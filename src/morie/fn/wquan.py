# morie.fn -- tail3 batch (rootcoder007/morie)
"""Weighted quantile, Harrell-Davis and inverse-weighted-ECDF forms.

Source consulted: Harrell, F.E. & Davis, C.E. (1982). A new distribution-free
quantile estimator.  *Biometrika* 69(3), 635-640.  Their estimator is the
linear combination of order statistics

    Q_p = sum_i W_{n,i} X_(i),
    W_{n,i} = I_{i/n}(p(n+1), (1-p)(n+1)) - I_{(i-1)/n}(p(n+1), (1-p)(n+1))

with I_x(a, b) the regularized incomplete beta function (their equations (2)
and (3)).  The weighted generalisation replaces the equally spaced knots
i/n by the cumulative normalised observation weights, so that an observation
carrying weight w contributes a beta-probability slab of width w; with equal
weights it reduces exactly to Harrell and Davis.  The plain inverse weighted
empirical CDF, q_p = inf{y : F_w(y) >= p}, is reported alongside it.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["weighted_quantile"]


def weighted_quantile(y, weights=None, p=0.5):
    """Weighted quantile of ``y`` at probability ``p``.

    Parameters
    ----------
    y : array-like
        Observations.
    weights : array-like, optional
        Non-negative observation weights; equal weights if omitted.
    p : float
        Probability in (0, 1).

    Returns
    -------
    RichResult
        estimate (Harrell-Davis, weighted), hd, ecdf (inverse weighted ECDF),
        w (order-statistic weights), p, n, method.

    References
    ----------
    Harrell & Davis (1982), Biometrika 69(3), 635-640, eq. (2)-(3).
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    n = int(yy.size)
    if weights is None:
        wv = [1.0] * n
    else:
        wv = [float(v) for v in np.atleast_1d(np.asarray(weights, dtype=float)).ravel()][:n]
    order = list(np.argsort(yy))
    xs = [float(yy[int(k)]) for k in order]
    ws = [wv[int(k)] for k in order]
    tot = sum(ws)
    pf = float(p)
    a = pf * (n + 1.0)
    b = (1.0 - pf) * (n + 1.0)
    knots = [0.0]
    run = 0.0
    for v in ws:
        run += v / tot
        knots.append(run)
    knots[n] = 1.0
    wt = []
    prev = float(stats.beta.cdf(0.0, a, b))
    for i in range(n):
        cur = float(stats.beta.cdf(knots[i + 1], a, b))
        wt.append(cur - prev)
        prev = cur
    hd = sum(wt[i] * xs[i] for i in range(n))
    cum = 0.0
    ecdf = xs[n - 1]
    for i in range(n):
        cum += ws[i] / tot
        if cum >= pf:
            ecdf = xs[i]
            break
    return RichResult(
        payload={
            "estimate": float(hd),
            "hd": float(hd),
            "ecdf": float(ecdf),
            "w": np.asarray(wt, dtype=float),
            "p": pf,
            "n": n,
            "method": "Harrell-Davis weighted quantile (Harrell & Davis 1982)",
        }
    )


# CANONICAL TEST
# >>> # weights all equal: the estimator is the plain Harrell-Davis quantile,
# >>> # and for a symmetric sample the median lands on the centre point
# >>> r = weighted_quantile([1.0, 2.0, 3.0], None, 0.5)
# >>> assert abs(r["estimate"] - 2.0) < 1e-12
# >>> assert abs(float(np.sum(r["w"])) - 1.0) < 1e-12


def cheatsheet():
    return "wquan(y, weights, p): Harrell-Davis weighted quantile."
