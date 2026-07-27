# morie.fn -- function file (rootcoder007/morie)
"""Manski monotone instrumental variable bounds."""

import numpy as np

from ._richresult import RichResult

__all__ = ["monotone_iv_bounds"]


def monotone_iv_bounds(y, D, Z, y_min=None, y_max=None):
    r"""Manski-Pepper MIV bounds on the mean potential outcomes.

    Without an exclusion restriction the worst-case (Manski) bound on
    :math:`E[Y(d)]` conditional on an instrument level v is

    .. math:: E[Y \mid D=d, Z=v] P(D=d \mid Z=v)
              + y_{\min} P(D \ne d \mid Z=v)
              \;\le\; E[Y(d) \mid Z=v] \;\le\;
              \ldots + y_{\max} P(D \ne d \mid Z=v),

    replacing the unobserved arm by its logical extremes. The
    *monotone* IV assumption says :math:`E[Y(d) \mid Z=v]` is
    nondecreasing in v, which lets the bounds be sharpened by taking

    .. math:: \max_{v' \le v} LB(v'), \qquad \min_{v'' \ge v} UB(v''),

    the intersection over ordered instrument levels. The reported
    bounds average the sharpened level-wise bounds over the observed
    distribution of Z.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    Z : array-like, shape (n,)
        Ordered monotone instrument (any orderable values).
    y_min, y_max : float, optional
        Logical outcome bounds; default the observed min and max.

    Returns
    -------
    RichResult
        keys: ``lower``, ``upper`` (on the ATE), ``width``,
        ``ey1_bounds``, ``ey0_bounds``, ``worst_case`` (the Manski
        bounds ignoring monotonicity, for comparison), ``levels``,
        ``n``, ``method``.

    References
    ----------
    Manski, C. F. & Pepper, J. V. (2000). Monotone instrumental
    variables: with an application to the returns to schooling.
    *Econometrica*, 68(4), 997-1010.

    Manski, C. F. (1990). Nonparametric bounds on treatment effects.
    *American Economic Review*, 80(2), 319-323.
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    Z = np.asarray(Z).ravel()
    n = y.size
    if not (D.size == n and Z.size == n):
        raise ValueError("y, D, Z must have equal length.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    lo_y = float(y.min()) if y_min is None else float(y_min)
    hi_y = float(y.max()) if y_max is None else float(y_max)
    if hi_y <= lo_y:
        raise ValueError("y_max must exceed y_min.")

    levels = np.unique(Z)
    if levels.size < 2:
        raise ValueError("need at least 2 instrument levels.")

    def level_bounds(d):
        lb, ub, wt = [], [], []
        for v in levels:
            sel = Z == v
            arm = sel & (D == d)
            p = arm.sum() / sel.sum()
            mean = y[arm].mean() if arm.any() else 0.0
            lb.append(mean * p + lo_y * (1 - p))
            ub.append(mean * p + hi_y * (1 - p))
            wt.append(sel.sum() / n)
        return np.array(lb), np.array(ub), np.array(wt)

    out = {}
    worst = {}
    for d in (1, 0):
        lb, ub, wt = level_bounds(d)
        worst[d] = (float(lb @ wt), float(ub @ wt))
        # MIV sharpening: monotone in the level order
        lb_s = np.maximum.accumulate(lb)
        ub_s = np.minimum.accumulate(ub[::-1])[::-1]
        out[d] = (float(lb_s @ wt), float(ub_s @ wt))

    lower = out[1][0] - out[0][1]
    upper = out[1][1] - out[0][0]
    return RichResult(
        payload={
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "ey1_bounds": out[1],
            "ey0_bounds": out[0],
            "worst_case": (worst[1][0] - worst[0][1], worst[1][1] - worst[0][0]),
            "levels": levels,
            "n": int(n),
            "method": "Manski-Pepper MIV bounds on the ATE (level-wise sharpening)",
        }
    )


def cheatsheet():
    return "mivbnd: worst-case arms filled with y_min/y_max, sharpened monotonically in Z"
