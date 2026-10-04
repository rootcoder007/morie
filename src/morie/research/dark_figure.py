# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P1: the dark figure of crime as a partial-identification problem.

Every identity and bound here is a machine-checked theorem in
``research/lean/P1DarkFigure.lean`` (Lean 4 + Mathlib, 0 sorry, standard
axioms only), building on ``P5Fairness.lean``:

* ``Research.P1.TwoSource.petersen_identity``: N = theta n1 n2 / m
* ``Research.P1.TwoSource.lincoln_petersen``: theta = 1 gives N = n1 n2 / m
* ``Research.P1.TwoSource.petersen_bounds``: theta in [1/kappa, kappa] gives
  n1 n2 / (kappa m) <= N <= kappa n1 n2 / m, both ends attained
* ``Research.P1.true_rate_bounds`` / ``dark_figure_bounds``
* ``Research.P1.conclusion_holds_below_breakdown`` / ``conclusion_fails_above_breakdown``
* ``Research.P1.three_list_saturated_fits`` / ``missing_cell_unconstrained`` /
  ``petersen_ge_floor`` / ``chapman_ge_floor``
* ``Research.P1.offence_count_bounds`` / ``offence_count_eq`` / ``category_not_identified``

The theorems are about expected counts and rates. Whether two lists are
independent, or what the misreporting boxes are, is a claim about the world
that no proof supplies; the functions take those as arguments.

R parity: ``rmorie`` ``R/dark_figure.R`` (``morie_dark_figure_*``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = [
    "dark_figure_two_source",
    "dark_figure_bounds",
    "dark_figure_breakdown",
    "dark_figure_three_list",
    "dark_figure_hierarchy",
]


def _scalar(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or v != v:
        raise ValueError(f"{name} must be a single non-missing number")
    return float(v)


def dark_figure_two_source(n1, n2, m, kappa=1.0, chapman=False) -> dict:
    """Two-source (capture-recapture) count with a dependence box.

    Under independence the true count is the Lincoln-Petersen ratio ``n1 n2 / m``.
    With the dependence factor only known to lie in ``[1/kappa, kappa]`` the true
    count lies in ``[n1 n2 / (kappa m), kappa n1 n2 / m]``, both ends attainable
    (``Research.P1.TwoSource.petersen_bounds``).

    Examples
    --------
    >>> dark_figure_two_source(400, 250, 80)["point"]
    1250.0
    >>> r = dark_figure_two_source(400, 250, 80, kappa=2)
    >>> (r["lower"], r["upper"])
    (625.0, 2500.0)
    """
    n1 = _scalar(n1, "n1")
    n2 = _scalar(n2, "n2")
    m = _scalar(m, "m")
    kappa = _scalar(kappa, "kappa")
    if n1 <= 0 or n2 <= 0 or m <= 0:
        raise ValueError("n1, n2 and m must be positive")
    if m > min(n1, n2):
        raise ValueError("m cannot exceed the smaller list")
    if kappa < 1:
        raise ValueError("kappa must be at least 1")
    point = n1 * n2 / m
    out = {
        "point": point,
        "lower": point / kappa,
        "upper": kappa * point,
        "kappa": kappa,
        "theorem": "Research.P1.TwoSource.lincoln_petersen" if kappa == 1 else "Research.P1.TwoSource.petersen_bounds",
    }
    if chapman:
        out["chapman"] = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    return out


def dark_figure_bounds(v_obs, r, alpha_max, beta_max):
    """Sharp bounds on a true victimisation rate and its dark figure.

    ``max(r, (v_obs - a)/(1 - a)) <= v <= v_obs/(1 - b)`` with the boxes
    ``alpha in [0, a]``, ``beta in [0, b]``, ``a + b < 1``
    (``Research.P1.true_rate_bounds``, ``dark_figure_bounds``). Returns a frame
    with ``v_obs, r, v_lower, v_upper, dark_lower, dark_upper, ratio_upper``.

    Examples
    --------
    >>> b = dark_figure_bounds(0.06, 0.02, alpha_max=0.01, beta_max=0.30)
    >>> round(float(b["v_upper"][0]), 10)
    0.0857142857
    """
    v = np.atleast_1d(np.asarray(v_obs, dtype=float))
    if np.any(np.isnan(v)) or np.any(v < 0) or np.any(v > 1):
        raise ValueError("v_obs must be numeric in [0, 1]")
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(np.isnan(rr)) or np.any(rr < 0) or np.any(rr > 1):
        raise ValueError("r must be numeric in [0, 1]")
    if rr.shape[0] != 1 and rr.shape[0] != v.shape[0]:
        raise ValueError("r must have length 1 or the length of v_obs")
    a = _scalar(alpha_max, "alpha_max")
    b = _scalar(beta_max, "beta_max")
    if a < 0 or b < 0:
        raise ValueError("alpha_max and beta_max must be single non-negative numbers")
    if a + b >= 1:
        raise ValueError("alpha_max + beta_max must be below 1")
    if rr.shape[0] == 1:
        rr = np.full(v.shape[0], float(rr[0]))
    v_lower = np.maximum(rr, (v - a) / (1 - a))
    v_upper = np.minimum(1.0, v / (1 - b))
    v_upper = np.maximum(v_upper, v_lower)
    ratio = np.where(rr > 0, v_upper / np.where(rr > 0, rr, 1.0), np.nan)
    return pd.DataFrame(
        {
            "v_obs": v,
            "r": rr,
            "v_lower": v_lower,
            "v_upper": v_upper,
            "dark_lower": v_lower - rr,
            "dark_upper": v_upper - rr,
            "ratio_upper": ratio,
        }
    )


def dark_figure_breakdown(v_obs, threshold) -> dict:
    """Breakdown analysis for a dark-figure conclusion.

    "The true rate is below ``threshold``" survives every admissible noise pair
    while the under-reporting box stays below ``1 - v_obs/threshold``
    (``Research.P1.conclusion_holds_below_breakdown``) and fails at or above it
    (``conclusion_fails_above_breakdown``).

    Examples
    --------
    >>> round(dark_figure_breakdown(0.06, threshold=0.10)["breakdown_beta_max"], 12)
    0.4
    """
    v = _scalar(v_obs, "v_obs")
    t = _scalar(threshold, "threshold")
    if v <= 0 or v > 1:
        raise ValueError("v_obs must be a single number in (0, 1]")
    if t <= v:
        raise ValueError("threshold must be a single number above v_obs")
    return {
        "breakdown_beta_max": 1 - v / t,
        "ratio": t / v,
        "theorems": ["Research.P1.conclusion_holds_below_breakdown", "Research.P1.conclusion_fails_above_breakdown"],
    }


_CELLS = ("100", "010", "001", "110", "101", "011", "111")


def dark_figure_three_list(counts, candidate_missing=None) -> dict:
    """Three-list capture-recapture: what is and is not identified.

    The saturated model reproduces any positive eight-cell table
    (``Research.P1.three_list_saturated_fits``), so the missing cell is free
    (``missing_cell_unconstrained``); setting the three-way interaction to zero
    gives ``m000 = m111 m100 m010 m001 / (m110 m101 m011)``. Pairwise Petersen
    and Chapman estimates respect the floor (``petersen_ge_floor``, ``chapman_ge_floor``).

    Examples
    --------
    >>> r = dark_figure_three_list({"100": 120, "010": 90, "001": 70, "110": 40, "101": 30, "011": 25, "111": 15})
    >>> (r["observed"], round(r["missing_no_three_way"], 6))
    (390.0, 378.0)
    """
    if not all(k in counts for k in _CELLS):
        raise ValueError("counts must be named by the seven cells 100, 010, 001, 110, 101, 011, 111")
    m = {k: float(counts[k]) for k in _CELLS}
    if any(v != v or v <= 0 for v in m.values()):
        raise ValueError("all seven observed cells must be positive")
    observed = sum(m.values())
    m000 = m["111"] * m["100"] * m["010"] * m["001"] / (m["110"] * m["101"] * m["011"])
    on = lambda cell, k: cell[k - 1] == "1"  # noqa: E731
    rows = []
    for a, b in ((1, 2), (1, 3), (2, 3)):
        n1 = sum(m[c] for c in _CELLS if on(c, a))
        n2 = sum(m[c] for c in _CELLS if on(c, b))
        mm = sum(m[c] for c in _CELLS if on(c, a) and on(c, b))
        rows.append(
            {
                "lists": f"{a}-{b}",
                "n1": n1,
                "n2": n2,
                "m": mm,
                "floor": n1 + n2 - mm,
                "petersen": n1 * n2 / mm,
                "chapman": (n1 + 1) * (n2 + 1) / (mm + 1) - 1,
            }
        )
    pairwise = pd.DataFrame(rows)
    implied = None
    if candidate_missing is not None:
        cm = np.atleast_1d(np.asarray(candidate_missing, dtype=float))
        if np.any(cm <= 0):
            raise ValueError("candidate_missing must be positive")
        lm = {k: math.log(v) for k, v in m.items()}
        three = (lm["111"] - lm["110"] - lm["101"] - lm["011"] + lm["100"] + lm["010"] + lm["001"]) - np.log(cm)
        implied = pd.DataFrame({"missing": cm, "N": observed + cm, "three_way": three})
    return {
        "observed": observed,
        "missing_no_three_way": m000,
        "N_no_three_way": observed + m000,
        "pairwise": pairwise,
        "implied_three_way": implied,
        "theorems": [
            "Research.P1.three_list_saturated_fits",
            "Research.P1.missing_cell_unconstrained",
            "Research.P1.petersen_ge_floor",
            "Research.P1.chapman_ge_floor",
        ],
    }


def dark_figure_hierarchy(offences_per_incident) -> dict:
    """Incident versus offence counting: the hierarchy-rule arithmetic.

    With N incidents carrying k_i >= 1 offences, bounded by K, the offence count
    lies in [N, KN] and equals N(1 + mean extra) (``Research.P1.offence_count_bounds``,
    ``offence_count_eq``); it is not a function of N (``category_not_identified``).

    Examples
    --------
    >>> r = dark_figure_hierarchy([1, 1, 2, 1, 3, 1, 1, 2])
    >>> (r["incidents"], r["offences"], r["mean_extra"])
    (8, 12, 0.5)
    """
    k = [int(v) for v in np.atleast_1d(np.asarray(offences_per_incident))]
    if len(k) == 0 or any(v < 1 for v in k):
        raise ValueError("every incident must carry at least one offence")
    n = len(k)
    tot = sum(k)
    return {
        "incidents": n,
        "offences": tot,
        "ratio": tot / n,
        "mean_extra": sum(v - 1 for v in k) / n,
        "bounds": {"lower": n, "upper": max(k) * n},
        "theorems": [
            "Research.P1.offence_count_bounds",
            "Research.P1.offence_count_eq",
            "Research.P1.category_not_identified",
        ],
    }
