# morie.fn -- function file (rootcoder007/morie)
"""Doubly robust DiD over a staggered adoption design: the ATT(g, t) table.

Where drcef collapses the group-time effects to a single curve in event
time, this module reports the table itself: one doubly robust estimate
per (cohort g, event time e) cell that the panel can support.

Sources opened: Callaway, B. and Sant'Anna, P. H. C. (2021), *Journal of
Econometrics* 225(2), 200-230 (arXiv:1803.09015), which defines
ATT(g, t) = E[Y_t(g) - Y_t(0) | G_g = 1] and estimates each cell against
the never-treated comparison group with period g - 1 as the base period;
and Sant'Anna, P. H. C. and Zhao, J. (2020), *Journal of Econometrics*
219(1), 101-122, equation (2.6), the doubly robust moment used for the
cell.  The staggered-rollout literature this design belongs to is Roth,
J. and Sant'Anna, P. H. C. (2023), *Journal of Political Economy
Microeconomics* 1(4), 669-709, and Borusyak, K., Jaravel, X. and Spiess,
J. (2024), *Review of Economic Studies*.

Pre-treatment cells (e < 0) are computed on identical footing and
returned rather than suppressed: their being near zero is the parallel
trends evidence.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_staggered_design"]


def dr_staggered_design(y, D, unit, time, cohort, X=None):
    """DR ATT for every (cohort, event-time) cell of a staggered rollout.

    Parameters
    ----------
    y : array-like
        Outcome, long format, one entry per unit-period.
    D : array-like
        Treatment indicator, carried for the interface.
    unit, time : array-like
        Unit and period identifiers.
    cohort : array-like
        First treated period of the row's unit; 0 or inf marks a
        never-treated unit.
    X : 2-D array-like, optional
        Baseline covariates, one row per unit-period.

    Returns
    -------
    result : dict
        Keys: estimate (cohort-size-weighted mean over post cells),
        cohorts, event_time, att (row-major cohort x event-time),
        n_cells, n_post, n.

    References
    ----------
    Callaway & Sant'Anna (2021), J. Econometrics 225(2):200-230.
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
    Roth & Sant'Anna (2023), JPE Microeconomics 1(4):669-709.
    """
    yv = k.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    u = [str(x) for x in unit]
    t = [float(x) for x in time]
    g = [float(x) for x in cohort]
    if not (len(u) == n and len(t) == n and len(g) == n):
        raise ValueError("y, unit, time and cohort must have the same length")
    Xr = k.mat(X) if X is not None else None
    val, gof, xof = {}, {}, {}
    for i in range(n):
        val[(u[i], t[i])] = yv[i]
        gof[u[i]] = g[i]
        if Xr is not None:
            xof[u[i]] = Xr[i]
    units = []
    for x in u:
        if x not in units:
            units.append(x)
    per = sorted(set(t))
    cohorts = sorted(set(gof[z] for z in units
                         if gof[z] > 0.0 and gof[z] != float("inf")))
    if not cohorts:
        raise ValueError("no treated cohort in the panel")
    never = [z for z in units if not (gof[z] > 0.0 and gof[z] != float("inf"))]
    if not never:
        raise ValueError("no never-treated units to serve as comparison")
    lo = int(round(per[0] - cohorts[-1]))
    hi = int(round(per[-1] - cohorts[0]))
    es = list(range(lo, hi + 1))
    att = []
    ncell = []
    num, den = 0.0, 0.0
    npost = 0
    for c in cohorts:
        size = float(sum(1 for z in units if gof[z] == c))
        for e in es:
            dys, ds, xs = [], [], []
            if (c - 1.0) in per and (c + e) in per:
                for z in units:
                    key1, key0 = (z, c + e), (z, c - 1.0)
                    if key1 not in val or key0 not in val:
                        continue
                    if gof[z] == c:
                        dys.append(val[key1] - val[key0])
                        ds.append(1.0)
                    elif z in never:
                        dys.append(val[key1] - val[key0])
                        ds.append(0.0)
                    else:
                        continue
                    if Xr is not None:
                        xs.append(xof[z])
            if len(dys) < 3 or sum(ds) <= 0.0 or sum(ds) >= len(ds):
                att.append(float("nan"))
                ncell.append(len(dys))
                continue
            fit = k.drdid_panel(dys, ds, xs if Xr is not None else None)
            att.append(fit["tau"])
            ncell.append(len(dys))
            if e >= 0:
                num += size * fit["tau"]
                den += size
                npost += 1
    return RichResult(
        title="Staggered DR-DiD",
        summary_lines=[("cohorts", len(cohorts)), ("horizons", len(es))],
        payload={
            "estimate": (num / den) if den > 0.0 else float("nan"),
            "cohorts": cohorts,
            "event_time": es,
            "att": att,
            "n_cells": ncell,
            "n_post": npost,
            "n": n,
            "method": "DR-DiD for staggered adoption",
        },
    )


def cheatsheet():
    return "drsta: DR-DiD for staggered adoption"
