# morie.fn -- function file (rootcoder007/morie)
"""Callaway-Sant'Anna event-study aggregation of DR group-time ATTs.

Sources opened: Callaway, B. and Sant'Anna, P. H. C. (2021).
Difference-in-differences with multiple time periods.  *Journal of
Econometrics* 225(2), 200-230; working paper arXiv:1803.09015, Section
3.4, equation (3.4),

    theta_es(e) = sum_g 1{g + e <= T} P(G = g | G + e <= T) ATT(g, g + e),

i.e. the group-time effects at a fixed length of exposure e = t - g are
averaged with the cohort-share weights of the cohorts that are actually
observed at that horizon.  Each ATT(g, g + e) is the doubly robust
panel estimator of Sant'Anna, P. H. C. and Zhao, J. (2020), *Journal of
Econometrics* 219(1), 101-122, equation (2.6), taken between period
g - 1 (base) and period g + e, with the never-treated units as
comparison group.

The difference from an unweighted event study is the weight: at horizon
e the late cohorts drop out of the sample, and equation (3.4)
renormalises over the survivors instead of letting the composition of
the average drift.
"""

from __future__ import annotations

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["dr_callaway_event_study"]


def dr_callaway_event_study(y, D, unit, time, cohort, X=None):
    """Event-study aggregation theta_es(e) of the DR ATT(g, t).

    Parameters
    ----------
    y : array-like
        Outcome, long format, one entry per unit-period.
    D : array-like
        Treatment indicator; carried for the interface, the cohort
        labels are what identify treatment timing.
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
        Keys: estimate (mean of theta_es over e >= 0), event_time,
        theta, n_cohorts, weight_sum, n.

    References
    ----------
    Callaway & Sant'Anna (2021), J. Econometrics 225(2):200-230, eq. (3.4).
    Sant'Anna & Zhao (2020), J. Econometrics 219(1):101-122, eq. (2.6).
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
    T = per[-1]
    cohorts = sorted(set(gof[z] for z in units
                         if gof[z] > 0.0 and gof[z] != float("inf")))
    if not cohorts:
        raise ValueError("no treated cohort in the panel")
    never = [z for z in units if not (gof[z] > 0.0 and gof[z] != float("inf"))]
    if not never:
        raise ValueError("no never-treated units to serve as comparison")
    size = {}
    for c in cohorts:
        size[c] = float(sum(1 for z in units if gof[z] == c))
    es = []
    for c in cohorts:
        for p in per:
            if c - 1.0 in per:
                e = int(round(p - c))
                if e not in es:
                    es.append(e)
    es = sorted(es)
    ev, theta, wsum = [], [], []
    for e in es:
        elig = [c for c in cohorts
                if (c + e) <= T and (c + e) in per and (c - 1.0) in per]
        if not elig:
            continue
        tot = sum(size[c] for c in elig)
        acc = 0.0
        ok = False
        for c in elig:
            dys, ds, xs = [], [], []
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
                continue
            fit = k.drdid_panel(dys, ds, xs if Xr is not None else None)
            acc += (size[c] / tot) * fit["tau"]
            ok = True
        if not ok:
            continue
        ev.append(e)
        theta.append(acc)
        wsum.append(tot)
    post = [theta[i] for i in range(len(ev)) if ev[i] >= 0]
    return RichResult(
        title="DR Callaway-Sant'Anna event study",
        summary_lines=[("horizons", len(ev))],
        payload={
            "estimate": k.mean(post) if post else float("nan"),
            "event_time": ev,
            "theta": theta,
            "n_cohorts": len(cohorts),
            "weight_sum": wsum,
            "n": n,
            "method": "DR Callaway-Sant'Anna event-study aggregation",
        },
    )


def cheatsheet():
    return "drcef: DR Callaway-Sant'Anna event-study aggregation"
