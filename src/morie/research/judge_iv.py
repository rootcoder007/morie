# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P14: judge-leniency designs (``research/lean/P14Instrument.lean``; Imbens & Angrist 1994).

* ``Research.P14.itt_decomposition`` / ``first_stage_decomposition``
* ``Research.P14.late_identification`` / ``wald_with_defiers`` / ``defiers_can_flip``

R parity: ``rmorie`` ``R/judge_iv.R`` (``morie_judge_iv_population``, ``morie_judge_iv``).
"""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["judge_iv_population", "judge_iv"]

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
