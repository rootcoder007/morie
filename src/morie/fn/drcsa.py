# morie.fn -- function file (rootcoder007/morie)
"""Doubly-robust group-time ATT (Callaway and Sant'Anna 2021)."""

from . import _array_core as np

from ._did import add_intercept, as_panel, first_treatment, logit_fit, \
    logit_predict, ols_fit
from ._richresult import RichResult

__all__ = ["dr_callaway_santanna"]


def _dr_cell(dY, treated, ctrl, Xu, trim):
    """One doubly-robust 2x2 cell: cohort g against a clean control set."""
    sub = treated | ctrl
    d = treated[sub].astype(float)
    Xs = Xu[sub]
    dy = dY[sub]
    if d.sum() < 1 or (1 - d).sum() < 1:
        return None
    beta, separated = logit_fit(Xs, d)
    p = logit_predict(Xs, beta)
    keep = p < trim
    if keep.sum() < 2 or d[keep].sum() < 1 or (1 - d[keep]).sum() < 1:
        return None
    c = (d == 0) & keep
    m0 = Xs @ ols_fit(Xs[c], dy[c])
    w1 = np.where(keep, d, 0.0)
    w0 = np.where(keep, (1 - d) * p / np.maximum(1 - p, 1e-12), 0.0)
    r = dy - m0
    a1 = float(np.sum(w1 * r) / np.sum(w1))
    a0 = float(np.sum(w0 * r) / np.sum(w0))
    infl_sub = (w1 * (r - a1) / w1.mean() - w0 * (r - a0) / w0.mean())
    infl = np.zeros(dY.size)
    infl[sub] = infl_sub
    return {
        "att": a1 - a0,
        "infl": infl,
        "n_treated": int(d.sum()),
        "n_control": int((1 - d).sum()),
        "separated": bool(separated),
        "n_trimmed": int((~keep).sum()),
    }


def dr_callaway_santanna(y, D, unit, time, cohort=None, X=None,
                         control="notyet", trim=0.995):
    r"""Group-time ATT with covariates, estimated doubly robustly.

    Callaway and Sant'Anna's estimator with conditional parallel
    trends. Each :math:`(g,t)` cell is its own two-period problem --
    the long difference :math:`Y_t - Y_{g-1}` for cohort :math:`g`
    against a clean control group -- so the Sant'Anna-Zhao doubly-robust
    moment applies cell by cell:

    .. math:: \widehat{ATT}^{dr}(g,t) =
              \frac{\mathbb{E}_n[w_1 (\Delta_{g,t} Y - \hat m_0(X))]}
                   {\mathbb{E}_n[w_1]}
              - \frac{\mathbb{E}_n[w_0 (\Delta_{g,t} Y - \hat m_0(X))]}
                     {\mathbb{E}_n[w_0]}.

    The propensity and outcome models are refitted PER CELL, not once
    for the panel. That is not inefficiency: the comparison group
    changes with :math:`(g,t)` under the not-yet-treated rule, so a
    single pooled propensity would be scoring units against a
    population they are not being compared to.

    Covariates only matter for identification if they predict the
    TREND. ``covariate_adjustment`` reports how far the doubly-robust
    estimate moved from the unadjusted one, which is the honest
    measure of how much the conditioning did.

    Parameters
    ----------
    y, D : array-like, shape (n,)
        Outcome and absorbing treatment in long format.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.
    cohort : array-like, optional
        First-treated period per observation; derived from ``D``
        otherwise.
    X : array-like, shape (n, p), optional
        Time-invariant covariates. Only the first period's value per
        unit is used, since a time-varying control could itself be an
        outcome of treatment.
    control : {'notyet', 'never'}
        Comparison group.
    trim : float
        Propensity trimming threshold within each cell.

    Returns
    -------
    RichResult
        ``estimate`` (overall ATT), ``se``, ``ci``, ``att_gt``,
        ``event``, ``pretrend``, ``covariate_adjustment``,
        ``n_cells``, ``cells_dropped``, ``overlap_warnings``.

    References
    ----------
    Callaway and Sant'Anna (2021), *Journal of Econometrics*
    225:200-230, section 4.
    Sant'Anna and Zhao (2020), *Journal of Econometrics* 219:101-122.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> nu, T = 60, 6
    >>> gv = np.where(np.arange(nu) < 20, 3.0,
    ...               np.where(np.arange(nu) < 40, 4.0, np.inf))
    >>> xu = rng.normal(size=nu)
    >>> unit = np.repeat(np.arange(nu), T)
    >>> time = np.tile(np.arange(T), nu)
    >>> g = np.repeat(gv, T)
    >>> D = (time >= g).astype(float)
    >>> y = np.repeat(xu, T) + time * (1 + 0.5 * np.repeat(xu, T)) + 2.0 * D
    >>> out = dr_callaway_santanna(y, D, unit, time, X=np.repeat(xu, T))
    >>> bool(abs(out["estimate"] - 2.0) < 1e-8)
    True
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
        lookup = {float(p): i for i, p in enumerate(periods)}
        g = np.array([lookup.get(float(v), np.inf) if v >= 0 else np.inf
                      for v in cm[:, 0]])
    if not np.isfinite(g).any():
        raise ValueError("no unit is ever treated.")

    n_u, T = Y.shape
    if X is None:
        Xu = np.ones((n_u, 1))
    else:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        if Xa.shape[0] == len(y):
            # long format: take each unit's first-period covariate row
            Xm = np.empty((n_u, Xa.shape[1]))
            for j in range(Xa.shape[1]):
                Xm[:, j] = as_panel(Xa[:, j], unit, time)[0][:, 0]
            Xu = add_intercept(Xm)
        elif Xa.shape[0] == n_u:
            Xu = add_intercept(Xa)
        else:
            raise ValueError(
                "X has %d rows; expected %d (long) or %d (one per unit)."
                % (Xa.shape[0], len(y), n_u)
            )

    cells, dropped, warn = {}, [], []
    for gg in np.unique(g[np.isfinite(g)]):
        gi = int(gg)
        if gi < 1:
            continue
        base = gi - 1
        treated = g == gg
        for t in range(T):
            if t == base:
                continue
            ctrl = (~np.isfinite(g)) if control == "never" else (g > max(t, gg))
            dY = Y[:, t] - Y[:, base]
            cell = _dr_cell(dY, treated, ctrl, Xu, float(trim))
            if cell is None:
                dropped.append((float(gg), float(t)))
                continue
            cell["post"] = bool(t >= gg)
            cell["rel"] = float(t - gg)
            cell["unadjusted"] = float(dY[treated].mean() - dY[ctrl].mean())
            if cell["separated"]:
                warn.append((float(gg), float(t)))
            cells[(float(gg), float(t))] = cell
    if not cells:
        raise ValueError(
            "no (g, t) cell had both treated and control units with overlap; "
            "check the adoption pattern and the covariates."
        )

    post = {k: v for k, v in cells.items() if v["post"]}
    if not post:
        raise ValueError("no post-treatment cell exists.")
    sizes = {k: float((g == k[0]).sum()) for k in post}
    tot = sum(sizes.values())
    est = sum(sizes[k] / tot * post[k]["att"] for k in post)
    unadj = sum(sizes[k] / tot * post[k]["unadjusted"] for k in post)
    infl = sum(sizes[k] / tot * post[k]["infl"] for k in post)
    se = float(np.sqrt(np.sum(infl**2)) / n_u)

    event = {}
    for rel in sorted({v["rel"] for v in post.values()}):
        keys = [k for k, v in post.items() if v["rel"] == rel]
        w = sum(sizes[k] for k in keys)
        e = sum(sizes[k] / w * post[k]["att"] for k in keys)
        i = sum(sizes[k] / w * post[k]["infl"] for k in keys)
        event[rel] = {"att": e,
                      "se": float(np.sqrt(np.sum(i**2)) / n_u)}

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": float(est),
            "se": se,
            "ci": (est - z * se, est + z * se),
            "att_gt": {k: v["att"] for k, v in cells.items()},
            "att_gt_unadjusted": {k: v["unadjusted"] for k, v in cells.items()},
            "event": event,
            "pretrend": {k: v["att"] for k, v in cells.items() if not v["post"]},
            "att_unadjusted": float(unadj),
            "covariate_adjustment": float(est - unadj),
            "adjustment_note": (
                "covariates change the estimate only if they predict the "
                "TREND; covariate_adjustment measures how much they did"
            ),
            "n_cells": len(cells),
            "cells_dropped": dropped,
            "overlap_warnings": warn,
            "n_trimmed_by_cell": {k: v["n_trimmed"] for k, v in cells.items()},
            "control_group": control,
            "per_cell_models": (
                "the propensity and outcome models are refitted per (g,t) "
                "because the comparison group itself changes with (g,t)"
            ),
            "n_units": int(n_u),
            "n_periods": int(T),
            "method": "Doubly-robust Callaway-Sant'Anna ATT(g,t)",
        }
    )


def cheatsheet():
    return (
        "drcsa: doubly-robust ATT(g,t) with per-cell propensity and outcome "
        "models, clean controls, event-study aggregation"
    )
