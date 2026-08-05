# morie.fn -- function file (rootcoder007/morie)
"""Mean residual life (mean excess) plot."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_mean_residual_life"]


def evt_mean_residual_life(x, u_grid=None):
    """
    Mean residual life plot

    Formula: e(u) = mean(X - u | X > u)

    Linear in u above a threshold where the GPD holds, with slope
    xi/(1-xi) and intercept sigma_u0/(1-xi); constant when xi = 0.
    That linearity is what the plot is read for.

    Parameters
    ----------
    x : array-like
        Sample.
    u_grid : array-like or None
        Thresholds.  None uses twenty equally spaced values from the
        minimum to the 90th percentile.

    Returns
    -------
    result : dict
        Keys: u, e_u, se, n_exceed, estimate (slope of e(u) on u), n.

    References
    ----------
    Davison & Smith (1990), JRSS B 52(3):393-442.
    """
    x = core.vec(x)
    n = len(x)
    if n < 2:
        raise ValueError("empty input: need at least two observations")
    if u_grid is None:
        lo = min(x)
        hi = core.quantile7(x, 0.9)
        u_grid = [lo + (hi - lo) * i / 19.0 for i in range(20)]
    else:
        u_grid = core.vec(u_grid)
    if not u_grid:
        raise ValueError("u_grid is empty")
    us, es, se, nex = [], [], [], []
    for u in u_grid:
        ex = [v - u for v in x if v > u]
        k = len(ex)
        if k < 2:
            continue
        m = sum(ex) / k
        v = sum((e - m) ** 2 for e in ex) / (k - 1)
        us.append(u)
        es.append(m)
        se.append(math.sqrt(v / k))
        nex.append(k)
    if len(us) < 2:
        raise ValueError("no threshold leaves two exceedances; grid too high")
    mu_u = sum(us) / len(us)
    mu_e = sum(es) / len(es)
    sxx = sum((v - mu_u) ** 2 for v in us)
    sxy = sum((us[i] - mu_u) * (es[i] - mu_e) for i in range(len(us)))
    slope = sxy / sxx if sxx > 0.0 else float("nan")
    return RichResult(payload={
        "u": us,
        "e_u": es,
        "se": se,
        "n_exceed": nex,
        "estimate": slope,
        "n": n,
        "method": "mean residual life (mean excess) plot",
    })


def cheatsheet():
    return "evmrlp: mean residual life plot"


# compact alias per ledger/NAMING.md
evtmeanresiduallife = evt_mean_residual_life
