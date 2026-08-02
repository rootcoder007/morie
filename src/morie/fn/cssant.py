# morie.fn -- function file (rootcoder007/morie)
"""Callaway-Sant'Anna group-time average treatment effects."""

from . import _array_core as np

from ._did import as_panel, first_treatment
from ._richresult import RichResult

__all__ = ["callaway_santanna", "group_time_att", "aggregate_att"]


def group_time_att(Y, g, control="notyet"):
    r"""Every :math:`ATT(g,t)` on a balanced panel, with its influence terms.

    For cohort :math:`g` and period :math:`t` the estimand is

    .. math:: ATT(g,t) = E\big[Y_t(1) - Y_t(0) \mid G = g\big],

    identified by a single 2x2 comparison between the base period
    :math:`g-1` and period :math:`t`, against a CLEAN control group.
    No already-treated unit ever appears as a control, which is
    exactly the property TWFE lacks.

    Returns a dict of ``(g, t) -> {att, n_treated, n_control, infl}``,
    where ``infl`` is the per-unit influence function used for the
    aggregation standard errors.
    """
    n, T = Y.shape
    out = {}
    for gg in np.unique(g[np.isfinite(g)]):
        gi = int(gg)
        if gi < 1:
            continue  # no pre-period exists for a period-0 adopter
        base = gi - 1
        treated = g == gg
        for t in range(T):
            if t == base:
                continue
            if control == "never":
                ctrl = ~np.isfinite(g)
            else:
                # not-yet-treated at max(t, g): never-treated units plus
                # cohorts that adopt strictly later than both periods
                ctrl = g > max(t, gg)
            if treated.sum() == 0 or ctrl.sum() == 0:
                continue
            dY = Y[:, t] - Y[:, base]
            mt = float(dY[treated].mean())
            mc = float(dY[ctrl].mean())
            infl = np.zeros(n)
            infl[treated] = (dY[treated] - mt) / treated.sum() * n
            infl[ctrl] = -(dY[ctrl] - mc) / ctrl.sum() * n
            out[(float(gg), float(t))] = {
                "att": mt - mc,
                "n_treated": int(treated.sum()),
                "n_control": int(ctrl.sum()),
                "post": bool(t >= gg),
                "rel": float(t - gg),
                "infl": infl,
            }
    return out


def aggregate_att(gt, g, n_units, weights_by="cohort_size"):
    """Aggregate :math:`ATT(g,t)` into overall, event-study and cohort views.

    The aggregation is a CHOICE, not a formality: the overall effect
    weights each cohort by its size, the event study by relative
    time, and they answer different questions. Standard errors come
    from the aggregated influence functions, so they account for the
    fact that the same units enter many :math:`(g,t)` cells.
    """
    post = {k: v for k, v in gt.items() if v["post"]}
    if not post:
        return {}
    sizes = {gg: float((g == gg).sum()) for gg, _ in post}
    tot = sum(sizes[gg] for gg, _ in post)
    if weights_by == "equal":
        w = {k: 1.0 / len(post) for k in post}
    else:
        w = {k: sizes[k[0]] / tot for k in post}

    def combine(keys, wts):
        s = sum(wts.values())
        if s <= 0:
            return np.nan, np.nan
        est = sum(wts[k] / s * post[k]["att"] for k in keys)
        infl = sum(wts[k] / s * post[k]["infl"] for k in keys)
        se = float(np.sqrt(np.sum(infl**2) / n_units**2))
        return est, se

    overall, overall_se = combine(list(post), w)

    event = {}
    for rel in sorted({v["rel"] for v in post.values()}):
        keys = [k for k, v in post.items() if v["rel"] == rel]
        sub = {k: w[k] for k in keys}
        e, s = combine(keys, sub)
        event[rel] = {"att": e, "se": s}

    cohort = {}
    for gg in sorted({k[0] for k in post}):
        keys = [k for k in post if k[0] == gg]
        sub = {k: 1.0 for k in keys}
        e, s = combine(keys, sub)
        cohort[gg] = {"att": e, "se": s}

    calendar = {}
    for t in sorted({k[1] for k in post}):
        keys = [k for k in post if k[1] == t]
        sub = {k: w[k] for k in keys}
        e, s = combine(keys, sub)
        calendar[t] = {"att": e, "se": s}

    return {
        "overall": overall,
        "overall_se": overall_se,
        "event": event,
        "cohort": cohort,
        "calendar": calendar,
    }


def callaway_santanna(y, D, unit, time, cohort=None, control="notyet"):
    r"""Group-time average treatment effects (Callaway and Sant'Anna 2021).

    The estimator that replaces the staggered TWFE regression. Rather
    than one coefficient it reports a separate

    .. math:: \widehat{ATT}(g,t) = \big(\bar Y_{g,t}
              - \bar Y_{g,g-1}\big)
              - \big(\bar Y_{C,t} - \bar Y_{C,g-1}\big)

    for each adoption cohort :math:`g` and period :math:`t`, using a
    clean control group :math:`C` -- never-treated units, or units
    not yet treated by :math:`\max(t, g)`. Because no already-treated
    unit is ever a control, no negative-weight comparison can enter,
    and the aggregate is a genuine weighted average of causal
    effects.

    The disaggregated estimates are the point. Cells with
    :math:`t < g` are PRE-treatment: they are not effects, they are
    the parallel-trends check, and this function reports them
    separately as ``pretrend`` rather than folding them into an
    average.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome in long format.
    D : array-like, shape (n,)
        Absorbing binary treatment. Ignored if ``cohort`` is given.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.
    cohort : array-like, optional
        First-treated period per observation, ``inf`` for
        never-treated. Derived from ``D`` when omitted.
    control : {'notyet', 'never'}
        Comparison group. ``'notyet'`` uses more data and is the
        default of Callaway and Sant'Anna; ``'never'`` is the
        stricter choice when later adopters may anticipate.

    Returns
    -------
    RichResult
        ``estimate`` (overall ATT), ``se``, ``ci``, ``att_gt``
        (dict keyed by ``(g, t)``), ``event`` (relative-time
        aggregation), ``cohort_att``, ``calendar``, ``pretrend``,
        ``pretrend_max_abs``, ``control_group``, ``n_cells``.

    References
    ----------
    Callaway and Sant'Anna (2021), *Journal of Econometrics*
    225:200-230.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(9), 8)
    >>> time = np.tile(np.arange(8), 9)
    >>> gv = np.repeat([3., 3., 3., 5., 5., 5., np.inf, np.inf, np.inf], 8)
    >>> D = (time >= gv).astype(float)
    >>> y = unit * 0.3 + time * 0.2 + 2.0 * D
    >>> round(callaway_santanna(y, D, unit, time)["estimate"], 8)
    2.0
    """
    if control not in ("notyet", "never"):
        raise ValueError("control must be 'notyet' or 'never'.")
    Y, units, periods = as_panel(y, unit, time)
    if cohort is None:
        g, _, _, _ = first_treatment(D, unit, time, units, periods)
    else:
        cm, _, _ = as_panel(np.where(np.isfinite(cohort), cohort, -1.0),
                            unit, time)
        if np.any(cm.max(axis=1) != cm.min(axis=1)):
            raise ValueError("cohort must be constant within a unit.")
        g = cm[:, 0]
        g = np.where(g < 0, np.inf, g)
        # map period labels to indices
        lookup = {float(p): i for i, p in enumerate(periods)}
        g = np.array([lookup.get(float(v), np.inf) if np.isfinite(v) else np.inf
                      for v in g])
    if not np.isfinite(g).any():
        raise ValueError("no unit is ever treated.")
    if control == "never" and np.isfinite(g).all():
        raise ValueError(
            "control='never' needs never-treated units and every unit is "
            "eventually treated; use control='notyet'."
        )

    gt = group_time_att(Y, g, control=control)
    if not gt:
        raise ValueError(
            "no (g, t) cell has both treated and control units; check that "
            "some cohort adopts after period 0 with a clean comparison group."
        )
    agg = aggregate_att(gt, g, len(units))
    pre = {k: v["att"] for k, v in gt.items() if not v["post"]}
    z = 1.959963984540054
    est, se = agg.get("overall", np.nan), agg.get("overall_se", np.nan)
    return RichResult(
        payload={
            "estimate": float(est),
            "se": float(se),
            "ci": (est - z * se, est + z * se),
            "att_gt": {k: v["att"] for k, v in gt.items()},
            "n_by_cell": {k: (v["n_treated"], v["n_control"]) for k, v in gt.items()},
            "event": agg.get("event", {}),
            "cohort_att": agg.get("cohort", {}),
            "calendar": agg.get("calendar", {}),
            "pretrend": pre,
            "pretrend_max_abs": float(max((abs(v) for v in pre.values()),
                                          default=0.0)),
            "pretrend_note": (
                "cells with t < g are not effects; they are the "
                "parallel-trends check and are reported separately"
            ),
            "control_group": control,
            "cohorts": np.unique(g[np.isfinite(g)]),
            "n_cells": len(gt),
            "n_units": int(len(units)),
            "n_periods": int(len(periods)),
            "clean_controls": (
                "no already-treated unit is ever used as a control, so no "
                "negative-weight comparison can enter the aggregate"
            ),
            "method": "Callaway-Sant'Anna (2021) group-time ATT(g,t)",
        }
    )


def cheatsheet():
    return (
        "cssant: Callaway-Sant'Anna ATT(g,t) with clean controls, plus "
        "event-study, cohort and calendar aggregations and a pre-trend check"
    )
