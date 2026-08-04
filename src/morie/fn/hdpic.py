# morie.fn -- slice s03 (rootcoder007/morie)
"""Highest posterior density credible interval.

Source consulted: Chen, M.-H. and Shao, Q.-M. (1999).  Monte Carlo
estimation of Bayesian credible and HPD intervals.  *Journal of
Computational and Graphical Statistics* 8(1), 69-92.  Their estimator
scans the sorted draws: with n draws and a target coverage 1 - alpha,
let j = floor((1 - alpha) n); the HPD interval is the pair

    ( theta_(i) , theta_(i + j) )   minimising  theta_(i + j) - theta_(i)

over i = 1, ..., n - j.  The 1999 JCGS paper is paywalled; the estimator
is quoted in its standard published form.

The interval is only the true HPD region when the posterior is
unimodal -- for a multimodal posterior the HPD region is a union of
intervals and this scan returns the shortest single interval covering
1 - alpha.  That limitation is stated rather than hidden, and the
equal-tailed interval is returned alongside so the two can be compared.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["highest_density_credible_interval"]


def highest_density_credible_interval(samples, alpha=0.05):
    """Shortest interval covering 1 - alpha of the draws.

    Returns
    -------
    estimate : the interval width
    lo, hi   : the HPD endpoints
    eq_lo, eq_hi : the equal-tailed endpoints, for comparison
    """
    v = sorted(k.vec(samples))
    n = len(v)
    a = float(alpha)
    j = int((1.0 - a) * n)
    if j < 1:
        j = 1
    if j > n - 1:
        j = n - 1
    best = 0
    width = v[j] - v[0]
    for i in range(1, n - j):
        w = v[i + j] - v[i]
        if w < width:
            width = w
            best = i
    return RichResult(
        title="Highest posterior density interval",
        summary_lines=[("lo", v[best]), ("hi", v[best + j])],
        payload={
            "estimate": width,
            "width": width,
            "lo": v[best],
            "hi": v[best + j],
            "eq_lo": k.quantile7(v, a / 2.0),
            "eq_hi": k.quantile7(v, 1.0 - a / 2.0),
            "n": n,
            "method": "Chen and Shao (1999) HPD interval scan; single-interval, valid for a unimodal posterior",
        },
    )


def cheatsheet():
    return "hdpic: Highest posterior density credible interval"
