# morie.fn -- function file (rootcoder007/morie)
"""Threshold choice by stability of the modified scale and shape."""

import math

from . import _s03core as core
from ._richresult import RichResult
from .evpot import evt_pot_fit

__all__ = ["evt_threshold_select_lvar"]


def evt_threshold_select_lvar(x, u_grid=None, window=3):
    """
    Threshold selection by local variance of the GPD estimates

    Formula: argmin_u Var(log sigma*_u, xi_u)

    A GPD fitted above u0 implies, for any higher u, sigma_u =
    sigma_u0 + xi (u - u0), so the MODIFIED scale sigma*_u = sigma_u -
    xi u and the shape xi are both constant in u once the model holds.
    The threshold chosen is the one whose forward window of fits has the
    smallest combined variance of log sigma* and xi -- the first place
    the parameters stop drifting.

    Parameters
    ----------
    x : array-like
        Sample.
    u_grid : array-like or None
        Candidate thresholds.  None uses the 50th to 90th sample
        percentiles in nine steps.
    window : int
        Number of consecutive thresholds the variance is taken over.

    Returns
    -------
    result : dict
        Keys: u_star, score, estimate, u, scores, xi, mod_scale, n.

    References
    ----------
    Northrop & Coleman (2014), Extremes 17(2):289-303.
    """
    x = core.vec(x)
    n = len(x)
    if n < 10:
        raise ValueError("need at least ten observations to select a threshold")
    if u_grid is None:
        u_grid = [core.quantile7(x, 0.5 + 0.05 * i) for i in range(9)]
    else:
        u_grid = core.vec(u_grid)
    window = int(window)
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(u_grid) < window:
        raise ValueError("u_grid is shorter than the window")
    us, xis, mods = [], [], []
    for u in u_grid:
        if sum(1 for v in x if v > u) < 5:
            continue
        f = evt_pot_fit(x, u)
        us.append(u)
        xis.append(f["xi"])
        mods.append(f["modified_scale"])
    if len(us) < window:
        raise ValueError("too few usable thresholds after filtering")
    scores = []
    for i in range(len(us) - window + 1):
        xw = xis[i:i + window]
        mw = mods[i:i + window]
        if any(v <= 0.0 for v in mw):
            scores.append(float("inf"))
            continue
        lw = [math.log(v) for v in mw]
        scores.append(core.variance(lw, 1) + core.variance(xw, 1))
    best = 0
    for i in range(1, len(scores)):
        if scores[i] < scores[best]:
            best = i
    return RichResult(payload={
        "u_star": us[best],
        "score": scores[best],
        "estimate": us[best],
        "u": us,
        "scores": scores,
        "xi": xis,
        "mod_scale": mods,
        "n": n,
        "method": "threshold selection by local variance of GPD estimates",
    })


def cheatsheet():
    return "evtsthr: threshold selection by parameter stability"


# compact alias per ledger/NAMING.md
evtthresholdselectlvar = evt_threshold_select_lvar
