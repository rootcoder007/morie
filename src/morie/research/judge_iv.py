# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P14: judge-leniency designs (``research/lean/P14Instrument.lean``; Imbens & Angrist 1994).

* ``Research.P14.itt_decomposition`` / ``first_stage_decomposition``
* ``Research.P14.late_identification`` / ``wald_with_defiers`` / ``defiers_can_flip``

R parity: ``rmorie`` ``R/judge_iv.R`` (``morie_judge_iv_population``, ``morie_judge_iv``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["judge_iv_population", "judge_iv", "judge_slope_test"]

_THEOREMS = [
    "Research.P14.itt_decomposition",
    "Research.P14.first_stage_decomposition",
    "Research.P14.late_identification",
    "Research.P14.wald_with_defiers",
    "Research.P14.defiers_can_flip",
]


def _weights(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    return w / float(w.sum())


def judge_iv_population(d0, d1, y0, y1, weights=None) -> dict:
    """The exact decomposition of a binary-instrument design on a known population.

    Examples
    --------
    >>> r = judge_iv_population([0, 0, 1, 0, 1], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 1, 1, 1])
    >>> ({k: round(v, 12) for k, v in r["shares"].items()}, round(r["wald"], 12))
    ({'complier': 0.4, 'defier': 0.2, 'always': 0.2, 'never': 0.2}, 0.0)
    """
    d0 = np.asarray(d0, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    n = d0.shape[0]
    if d1.shape[0] != n or y0.shape[0] != n or y1.shape[0] != n:
        raise ValueError("d0, d1, y0 and y1 must have equal length")
    if not (np.all((d0 == 0) | (d0 == 1)) and np.all((d1 == 0) | (d1 == 1))):
        raise ValueError("d0 and d1 must be 0/1")
    w = _weights(weights, n)
    types = []
    for i in range(n):
        if d0[i] == 0 and d1[i] == 1:
            types.append("complier")
        elif d0[i] == 1 and d1[i] == 0:
            types.append("defier")
        elif d0[i] == 1:
            types.append("always")
        else:
            types.append("never")
    eff = y1 - y0
    shares = {}
    effects = {}
    for t in ("complier", "defier", "always", "never"):
        idx = [i for i in range(n) if types[i] == t]
        shares[t] = float(sum(w[i] for i in idx))
        effects[t] = float(sum(w[i] * eff[i] for i in idx) / shares[t]) if shares[t] > 0 else float("nan")
    yz1 = np.where(d1 == 1, y1, y0)
    yz0 = np.where(d0 == 1, y1, y0)
    itt = float(np.sum(w * yz1) - np.sum(w * yz0))
    fs = float(np.sum(w * d1) - np.sum(w * d0))
    return {
        "shares": shares,
        "effects": effects,
        "itt": itt,
        "first_stage": fs,
        "wald": itt / fs if fs != 0 else float("nan"),
        "late": effects["complier"],
        "monotone": shares["defier"] == 0,
        "ate": float(np.sum(w * eff)),
        "theorems": _THEOREMS,
    }


def judge_iv(z, d, y, weights=None, defier_share=(0, 0.05, 0.1), defier_effect=(0,)) -> dict:
    """Observational Wald ratio, the type shares under monotonicity, and the defier sensitivity.

    Examples
    --------
    >>> z = [0, 0, 0, 0, 1, 1, 1, 1]; d = [0, 0, 1, 0, 1, 1, 1, 0]; y = [1, 2, 5, 1, 6, 4, 5, 2]
    >>> r = judge_iv(z, d, y)
    >>> (round(r["first_stage"], 12), round(r["itt"], 12), round(r["wald"], 12))
    (0.5, 2.0, 4.0)
    """
    z = np.asarray(z, dtype=float)
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    n = z.shape[0]
    if d.shape[0] != n or y.shape[0] != n:
        raise ValueError("z, d and y must have equal length")
    if not (np.all((z == 0) | (z == 1)) and np.all((d == 0) | (d == 1))):
        raise ValueError("z and d must be 0/1")
    if not (np.any(z == 1) and np.any(z == 0)):
        raise ValueError("z must take both values")
    w = _weights(weights, n)

    def m(v, mask):
        return float(np.sum(w[mask] * v[mask]) / np.sum(w[mask]))

    fs = m(d, z == 1) - m(d, z == 0)
    itt = m(y, z == 1) - m(y, z == 0)
    wald = itt / fs if fs != 0 else float("nan")
    always = m(d, z == 0)
    never = 1 - m(d, z == 1)
    shares = {"complier": 1 - always - never, "always": always, "never": never}
    rows = []
    for de in defier_effect:
        for ds in defier_share:
            pc = shares["complier"] + ds
            implied = (wald * (pc - ds) + ds * de) / pc if pc > 0 and wald == wald else float("nan")
            rows.append(
                {
                    "defier_share": float(ds),
                    "defier_effect": float(de),
                    "complier_share": pc,
                    "implied_complier_effect": implied,
                }
            )
    return {
        "first_stage": fs,
        "itt": itt,
        "wald": wald,
        "shares_if_monotone": shares,
        "sensitivity": pd.DataFrame(rows),
        "theorems": [
            "Research.P14.late_identification",
            "Research.P14.wald_with_defiers",
            "Research.P14.first_stage_decomposition",
        ],
    }


def judge_slope_test(judge, d, y, weights=None, lo=None, hi=None) -> dict:
    """The many-judge slope test of monotonicity (Frandsen, Lefgren & Leslie 2023).

    ``Research.P14Slope.propensity_mono`` / ``outcome_diff`` / ``slope_bound`` / ``violation_refutes_monotonicity``.

    Examples
    --------
    >>> judge = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    >>> d = [0, 0, 0, 1, 1, 0,  0, 1, 1, 1, 0, 1,  1, 1, 1, 1, 1, 0]
    >>> y = [1, 0, 0, 1, 0, 0,  0, 1, 1, 0, 0, 1,  1, 1, 1, 1, 0, 1]
    >>> s = judge_slope_test(judge, d, y)
    >>> (s["judges"]["judge"], [round(p, 12) for p in s["judges"]["propensity"]], s["pairs"]["violation"], s["violations"])
    (['A', 'B', 'C'], [0.333333333333, 0.666666666667, 0.833333333333], [False, False, True], 1)
    >>> [round(v, 12) for v in s["pairs"]["late"]]
    [0.5, 1.0, 2.0]
    """
    judge = [str(j) for j in judge]
    dd = [float(x) for x in d]
    yy = [float(x) for x in y]
    n = len(judge)
    if len(dd) != n or len(yy) != n:
        raise ValueError("judge, d and y must have equal length")
    if any(math.isnan(x) for x in dd + yy):
        raise ValueError("no missing values allowed")
    if any(x not in (0.0, 1.0) for x in dd):
        raise ValueError("d must be 0/1")
    lo = min(yy) if lo is None else float(lo)
    hi = max(yy) if hi is None else float(hi)
    if any(v < lo or v > hi for v in yy):
        raise ValueError("y must lie in [lo, hi]")
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    if len(w) != n or any(math.isnan(x) or x < 0 for x in w):
        raise ValueError("weights must be non-negative")
    ids = list(dict.fromkeys(judge))
    P, Y, cnt = {}, {}, {}
    for j in ids:
        sel = [i for i in range(n) if judge[i] == j]
        tw = math.fsum(w[i] for i in sel)
        if tw <= 0:
            raise ValueError("every judge needs positive total weight")
        P[j] = math.fsum(w[i] * dd[i] for i in sel) / tw
        Y[j] = math.fsum(w[i] * yy[i] for i in sel) / tw
        cnt[j] = len(sel)
    order = sorted(ids, key=lambda j: P[j])
    judges = {
        "judge": order,
        "n": [cnt[j] for j in order],
        "propensity": [P[j] for j in order],
        "outcome": [Y[j] for j in order],
    }
    pairs = None
    if len(order) >= 2:
        pairs = {k: [] for k in ("j", "k", "dP", "dY", "bound", "violation", "late")}
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                j, k = order[a], order[b]
                dP = P[k] - P[j]
                dY = Y[k] - Y[j]
                bound = (hi - lo) * dP
                pairs["j"].append(j)
                pairs["k"].append(k)
                pairs["dP"].append(dP)
                pairs["dY"].append(dY)
                pairs["bound"].append(bound)
                pairs["violation"].append(abs(dY) > bound + 1e-12)
                pairs["late"].append(dY / dP if dP > 0 else math.nan)
    viol = 0 if pairs is None else sum(pairs["violation"])
    return {
        "judges": judges,
        "pairs": pairs,
        "violations": viol,
        "monotone_consistent": viol == 0,
        "lo": lo,
        "hi": hi,
        "theorems": [
            "Research.P14Slope.propensity_mono",
            "Research.P14Slope.outcome_diff",
            "Research.P14Slope.slope_bound",
            "Research.P14Slope.violation_refutes_monotonicity",
        ],
    }
