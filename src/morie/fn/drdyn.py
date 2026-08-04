# morie.fn -- slice s03 (rootcoder007/morie)
"""Dynamic (event-study) doubly robust DiD.

Sources consulted (BOTH FETCHED): Callaway, B. and Sant'Anna, P. H. C.
(2021).  Difference-in-differences with multiple time periods.
*Journal of Econometrics* 225(2), 200-230 (arXiv:1803.09015), which
defines the group-time average treatment effect

    ATT(g, t) = E[ Y_t(g) - Y_t(0) | G_g = 1 ]

and the event-time aggregation that averages ATT(g, g+e) over the
cohorts g observed at that horizon; and Sant'Anna, P. H. C. and Zhao, J.
(2020), *Journal of Econometrics* 219(1), 101-122 (arXiv:1812.01723),
whose equation (2.6) supplies the doubly robust estimator used for each
(g, t) cell, with the never-treated units as the comparison group and
period g-1 as the base period.

Negative horizons are pre-treatment placebo cells; they are computed on
exactly the same footing, which is the point of an event study -- their
being near zero is the evidence for parallel trends, so they are
reported rather than suppressed.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["dr_dynamic_did"]


def dr_dynamic_did(y, D=None, unit=None, time=None, cohort=None, horizon=3,
                   X=None):
    """ATT by event time, e = -horizon .. +horizon.

    Parameters
    ----------
    y : array-like
        Outcome, long format (one row per unit-period).
    D : array-like, optional
        Ignored when ``cohort`` is given; treatment indicator otherwise.
    unit, time : array-like
        Unit and period identifiers.
    cohort : array-like
        First treated period per row; 0 or inf marks never-treated.
    horizon : int
        Largest |event time| reported.
    X : 2-D array-like, optional
        Baseline covariates, one row per unit-period.

    Returns
    -------
    RichResult with payload:
        estimate  : the simple average of the post-treatment ATTs
        event_time: the horizons reported
        att       : ATT at each horizon
        n_cells   : units contributing at each horizon
    """
    yv = k.vec(y)
    u = [str(x) for x in unit]
    t = [float(x) for x in time]
    g = [float(x) for x in cohort]
    Xr = k.mat(X) if X is not None else None
    units = []
    for x in u:
        if x not in units:
            units.append(x)
    # y[unit][period]
    per = sorted(set(t))
    val = {}
    gof = {}
    xof = {}
    for i in range(len(yv)):
        val[(u[i], t[i])] = yv[i]
        gof[u[i]] = g[i]
        if Xr is not None:
            xof[u[i]] = Xr[i]
    hs = list(range(-int(horizon), int(horizon) + 1))
    att = []
    ncell = []
    post = []
    for e in hs:
        dys = []
        ds = []
        xs = []
        for uu in units:
            gg = gof[uu]
            treated = gg > 0.0 and gg != float("inf")
            for p in per:
                if treated:
                    if p != gg + e:
                        continue
                    base = gg - 1.0
                else:
                    base = None
                if treated:
                    if (uu, p) in val and (uu, base) in val:
                        dys.append(val[(uu, p)] - val[(uu, base)])
                        ds.append(1.0)
                        if Xr is not None:
                            xs.append(xof[uu])
            if not treated:
                # never-treated contribute the same calendar contrast as the
                # cohorts observed at this horizon
                for gg2 in sorted(set([gof[z] for z in units
                                       if gof[z] > 0.0 and gof[z] != float("inf")])):
                    p = gg2 + e
                    base = gg2 - 1.0
                    if (uu, p) in val and (uu, base) in val:
                        dys.append(val[(uu, p)] - val[(uu, base)])
                        ds.append(0.0)
                        if Xr is not None:
                            xs.append(xof[uu])
        if len(dys) < 3 or sum(ds) <= 0.0 or sum(ds) >= len(ds):
            att.append(float("nan"))
            ncell.append(len(dys))
            continue
        fit = k.drdid_panel(dys, ds, xs if Xr is not None else None)
        att.append(fit["tau"])
        ncell.append(len(dys))
        if e >= 0:
            post.append(fit["tau"])
    good = [x for x in post if x == x]
    return RichResult(
        title="Dynamic DR-DiD",
        summary_lines=[("horizons", len(hs))],
        payload={
            "estimate": k.mean(good) if good else float("nan"),
            "event_time": hs,
            "att": att,
            "n_cells": ncell,
            "method": "Event-time ATT(g, g+e) (Callaway and Sant'Anna 2021) estimated by DR-DiD (Sant'Anna and Zhao 2020)",
        },
    )


def cheatsheet():
    return "drdyn: Dynamic DR-DiD over event-time horizon"


drdynamicdid = dr_dynamic_did
