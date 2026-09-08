# morie.fn -- function file (rootcoder007/morie)
"""Sun-Abraham interaction-weighted difference in differences."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['iwdid', 'sun_abraham_did', 'sunabrahamdid']


def iwdid(y, unit, time, cohort, never=0):
    """Sun-Abraham interaction-weighted difference in differences.

    The two-way fixed effects event-study coefficient is a contaminated combination of cohort-specific effects when treatment timing is staggered: its weights are non-convex and it loads on relative periods other than its own. The fix is to estimate each CATT(e,l) separately and aggregate with weights that are cohort shares, so no weight can be negative. Here CATT(e,l) is the 2x2 DID against the never-treated cohort with l = -1 as the base period, which is the choice of pre-period and control cohort described in the paper's Section 4.2.


    Formula: nu_g = (1/|g|) sum_{l in g} sum_e CATT(e,l) Pr{E=e | E in [-l, T-l]}

    Parameters
    ----------
    y : array-like
        Outcome, one row per unit-period.
    unit : array-like
        Unit identifier.
    time : array-like
        Calendar period.
    cohort : array-like
        First treated period for the row's unit.
    never : scalar
        Value of ``cohort`` marking never-treated units.

    Returns
    -------
    RichResult
        ``event_time``, ``att``, ``overall``, ``cohorts``, ``n``.

    References
    ----------
    Sun and Abraham (2021), Estimating dynamic treatment effects in event
    studies with heterogeneous treatment effects, Journal of Econometrics
    225(3):175-199.  Equations (26) and (28) for the estimand and the IW
    estimator, Section 4.2 for the DID choice of pre-period and control
    cohort.  Verified against the paper.
    """
    y = C.vec(y)
    unit = list(unit); time = [float(t) for t in time]
    cohort = [float(c) for c in cohort]
    n = len(y)
    cell = {}
    for i in range(n):
        cell.setdefault((cohort[i], time[i]), []).append(y[i])
    cm = {k: sum(v) / len(v) for k, v in cell.items()}
    nev = float(never)
    treated = sorted({c for c in cohort if c != nev})
    size = {g: len({unit[i] for i in range(n) if cohort[i] == g}) for g in treated}
    evs = sorted({t - g for g in treated for t in set(time)})
    ev_out, att_out = [], []
    for e in evs:
        num, den = 0.0, 0.0
        for g in treated:
            keys = [(g, g + e), (g, g - 1.0), (nev, g + e), (nev, g - 1.0)]
            if all(k in cm for k in keys):
                catt = ((cm[keys[0]] - cm[keys[1]])
                        - (cm[keys[2]] - cm[keys[3]]))
                num += size[g] * catt
                den += size[g]
        if den > 0:
            ev_out.append(e); att_out.append(num / den)
    post = [a for e, a in zip(ev_out, att_out) if e >= 0]
    return RichResult(payload={
        "event_time": ev_out, "att": att_out,
        "overall": (sum(post) / len(post)) if post else float("nan"),
        "cohorts": treated, "n": n,
        "method": "Sun-Abraham interaction-weighted DID"})


sun_abraham_did = iwdid
sunabrahamdid = iwdid


def cheatsheet():
    return "sunabr: Sun-Abraham interaction-weighted difference in differences."
