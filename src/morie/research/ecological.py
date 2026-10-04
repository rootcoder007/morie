# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P12: ecological versus individual correlation
(``research/lean/P12Ecological.lean``; Freedman, Pisani & Purves ch. 9 sec. 4; Robinson 1950).

* ``Research.P12.within_orth``: the within-group residual is orthogonal to every group-level function
* ``Research.P12.cov_decomp`` / ``var_decomp``: exact between/within decompositions
* ``Research.P12.ecological_ge``: zero within covariance gives corr(x, y)^2 <= corr(group means)^2
* ``Research.P12.dd_bounds`` / ``Cells.ends_attained`` / ``dd_complement`` / ``dd_aggregate_bounds`` (``P12Bounds.lean``)

R parity: ``rmorie`` ``R/ecological.R`` (``morie_ecological_decompose``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["ecological_decompose", "ecological_bounds"]


def ecological_decompose(x, y, group) -> dict:
    """Between/within decomposition of a correlation across groups.

    Population (divide by n) conventions throughout, matching the Lean definitions.

    Examples
    --------
    >>> r = ecological_decompose([0, 2, 1, 3], [1, 3, 0, 2], ["a", "a", "b", "b"])
    >>> (round(r["corr_individual"], 12), r["corr_ecological"], r["sign_reversed"])
    (0.6, -1.0, True)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group = [str(g) for g in group]
    n = x.shape[0]
    if y.shape[0] != n or len(group) != n:
        raise ValueError("x, y and group must have equal length")
    if n < 2:
        raise ValueError("need at least two individuals")
    sums_x: dict = {}
    sums_y: dict = {}
    cnt: dict = {}
    for i, g in enumerate(group):
        sums_x[g] = sums_x.get(g, 0.0) + float(x[i])
        sums_y[g] = sums_y.get(g, 0.0) + float(y[i])
        cnt[g] = cnt.get(g, 0) + 1
    bx = np.array([sums_x[g] / cnt[g] for g in group])
    by = np.array([sums_y[g] / cnt[g] for g in group])
    wx = x - bx
    wy = y - by

    def pcov(a, b):
        return float(np.sum(a * b)) / n - float(np.mean(a)) * float(np.mean(b))

    cv = {"individual": pcov(x, y), "between": pcov(bx, by), "within": pcov(wx, wy)}
    vx = {"individual": pcov(x, x), "between": pcov(bx, bx), "within": pcov(wx, wx)}
    vy = {"individual": pcov(y, y), "between": pcov(by, by), "within": pcov(wy, wy)}
    ci = (
        cv["individual"] / math.sqrt(vx["individual"] * vy["individual"])
        if vx["individual"] > 0 and vy["individual"] > 0
        else float("nan")
    )
    ce = (
        cv["between"] / math.sqrt(vx["between"] * vy["between"])
        if vx["between"] > 0 and vy["between"] > 0
        else float("nan")
    )
    reversed_ = (not math.isnan(ci)) and (not math.isnan(ce)) and ci != 0 and ce != 0 and (ci > 0) != (ce > 0)
    return {
        "cov": cv,
        "var_x": vx,
        "var_y": vy,
        "corr_individual": ci,
        "corr_ecological": ce,
        "within_share_of_cov": cv["within"] / cv["individual"] if cv["individual"] != 0 else float("nan"),
        "bound_applies": abs(cv["within"]) < 1e-12 * max(1.0, abs(cv["individual"])),
        "sign_reversed": bool(reversed_),
        "theorems": [
            "Research.P12.within_orth",
            "Research.P12.cov_decomp",
            "Research.P12.var_decomp",
            "Research.P12.ecological_ge",
        ],
    }


def ecological_bounds(p, q, weights=None) -> dict:
    """Duncan-Davis bounds: what neighbourhood marginals say about an individual rate.

    ``max(0, (p+q-1)/p) <= P(y | x) <= min(1, q/p)``, both ends attained
    (``Research.P12.dd_bounds``, ``Cells.ends_attained``); the complement rate
    follows from ``q = p r + (1-p) r'`` (``dd_complement``); the aggregate rate
    inherits the ``m_g p_g``-weighted mean of the intervals (``dd_aggregate_bounds``).

    Examples
    --------
    >>> r = ecological_bounds([0.2, 0.5, 0.8], [0.1, 0.3, 0.6], weights=[1000, 2000, 500])
    >>> ([round(float(v), 12) for v in r["neighbourhoods"]["upper"]], round(r["aggregate"]["upper"], 12))
    ([0.5, 0.6, 0.75], 0.625)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = p.shape[0]
    if q.shape[0] != n:
        raise ValueError("p and q must have equal length")
    if np.any(np.isnan(p)) or np.any(np.isnan(q)) or np.any((p <= 0) | (p >= 1)) or np.any((q < 0) | (q > 1)):
        raise ValueError("p must lie in (0, 1) and q in [0, 1]")
    w = np.ones(n) if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
    if w.shape[0] != n or np.any(w <= 0):
        raise ValueError("weights must be positive, one per neighbourhood")
    lower = np.maximum(0.0, (p + q - 1) / p)
    upper = np.minimum(1.0, q / p)
    m = w * p
    agg_lo = float(np.sum(m * lower) / np.sum(m))
    agg_hi = float(np.sum(m * upper) / np.sum(m))
    return {
        "neighbourhoods": pd.DataFrame(
            {
                "p": p,
                "q": q,
                "lower": lower,
                "upper": upper,
                "width": upper - lower,
                "point_identified": np.abs(upper - lower) < 1e-12,
                "complement_lower": (q - p * upper) / (1 - p),
                "complement_upper": (q - p * lower) / (1 - p),
            }
        ),
        "aggregate": {"lower": agg_lo, "upper": agg_hi, "width": agg_hi - agg_lo},
        "theorems": [
            "Research.P12.pq_ge",
            "Research.P12.dd_bounds",
            "Research.P12.Cells.ends_attained",
            "Research.P12.dd_complement",
            "Research.P12.dd_aggregate_bounds",
        ],
    }
