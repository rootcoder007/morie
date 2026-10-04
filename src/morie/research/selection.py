# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P2, P6, P8: identification results on selection, the age-crime curve and deterrence
(``research/lean/P2Selection.lean``, ``P2Benchmark.lean``, ``P2RelativeRisk.lean``, ``P2Interracial.lean``,
``P2Collider.lean``, ``P6AgeCrime.lean``, ``P8Deterrence.lean``, ``P8Comparative.lean``, ``P8Necessity.lean``).

R parity: ``rmorie`` ``R/research_p268.R``.
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._sci_core import brentq

__all__ = [
    "disparity_exposure_bounds",
    "age_crime_aggregate",
    "deterrence_design_check",
    "disparity_benchmark",
    "relative_risk_from_or",
    "deterrence_response",
    "interracial_rates",
    "probability_of_necessity",
    "collider_arrest",
]


def disparity_exposure_bounds(y, m, gamma=1.0, reference=None):
    """Bounds on a police-outcome disparity when exposure is measured by a proxy
    (``Research.P2.rate_bounds``, ``disparity_bounds``, ``disparity_sign_identified``).

    ``y`` and ``m`` are dicts keyed by group.

    Examples
    --------
    >>> d = disparity_exposure_bounds({"A": 300, "B": 100}, {"A": 1000, "B": 1000}, gamma=1.5, reference="B")
    >>> ([round(v, 12) for v in d["ratio_proxy"]], [round(v, 12) for v in d["ratio_lower"]], list(d["direction_identified"]))
    ([3.0, 1.0], [1.333333333333, 0.444444444444], [True, False])
    """
    if set(y) != set(m):
        raise ValueError("y and m must be named vectors over the same groups")
    groups = list(y)
    yv = np.array([float(y[g]) for g in groups])
    mv = np.array([float(m[g]) for g in groups])
    if np.any(yv <= 0) or np.any(mv <= 0):
        raise ValueError("y and m must be positive")
    gamma = float(gamma)
    if gamma != gamma or gamma < 1:
        raise ValueError("gamma must be a single number >= 1")
    reference = groups[0] if reference is None else str(reference)
    if reference not in groups:
        raise ValueError("reference must be one of the groups")
    pr = yv / mv
    ratio = pr / pr[groups.index(reference)]
    return pd.DataFrame(
        {
            "group": groups,
            "y": yv,
            "m": mv,
            "proxy_rate": pr,
            "rate_lower": pr / gamma,
            "rate_upper": gamma * pr,
            "ratio_proxy": ratio,
            "ratio_lower": ratio / gamma**2,
            "ratio_upper": gamma**2 * ratio,
            "direction_identified": (ratio > gamma**2) | (ratio < 1 / gamma**2),
        }
    )


def age_crime_aggregate(shares, curves, ages=None):
    """Aggregate age-crime curve of a mixture of latent types (``Research.P6.aggregate_not_identifying``).

    Examples
    --------
    >>> a = age_crime_aggregate([0.5, 0.5], [[1, 3], [2, 2], [3, 1]], ages=[10, 11, 12])
    >>> list(a["aggregate"])
    [2.0, 2.0, 2.0]
    """
    C = np.asarray(curves, dtype=float)
    sh = np.asarray(shares, dtype=float)
    if C.ndim != 2 or sh.shape[0] != C.shape[1]:
        raise ValueError("one share per type column")
    if np.any(sh < 0) or abs(float(sh.sum()) - 1) > 1e-8:
        raise ValueError("shares must be non-negative and sum to 1")
    if ages is None:
        ages = list(range(1, C.shape[0] + 1))
    agg = C @ sh
    data = {"age": np.asarray(ages), "aggregate": agg}
    for j in range(C.shape[1]):
        data[f"type{j + 1}"] = C[:, j]
    out = pd.DataFrame(data)
    out.attrs["equivalent_single_type"] = agg
    out.attrs["theorems"] = ["Research.P6.aggregate_not_identifying", "Research.P6.invariance_not_necessary"]
    return out


def deterrence_design_check(p, s, c) -> dict:
    """Which deterrence partial effects a design can identify (``Research.P8.constant_dimension_not_identified``).

    Examples
    --------
    >>> r = deterrence_design_check(p=[0.1, 0.3, 0.5], s=[2, 2, 2], c=[30, 30, 30])
    >>> (r["rank"], r["identified"])
    (2, {'certainty': True, 'severity': False, 'celerity': False})
    """
    p = np.asarray(p, dtype=float)
    s = np.asarray(s, dtype=float)
    c = np.asarray(c, dtype=float)
    n = p.shape[0]
    if s.shape[0] != n or c.shape[0] != n:
        raise ValueError("p, s and c must have equal length")
    X = np.column_stack([np.ones(n), p, s, c])
    rk = int(np.linalg.matrix_rank(X))
    full = rk == 4
    names = ["certainty", "severity", "celerity"]
    identified = {k: full for k in names}
    if not full:
        for j in range(1, 4):
            keep = [i for i in range(4) if i != j]
            identified[names[j - 1]] = int(np.linalg.matrix_rank(X[:, keep])) < rk
    return {
        "rank": rk,
        "identified": identified,
        "n_regimes": n,
        "constant": {
            "certainty": len(set(p.tolist())) == 1,
            "severity": len(set(s.tolist())) == 1,
            "celerity": len(set(c.tolist())) == 1,
        },
        "theorem": "Research.P8.constant_dimension_not_identified",
    }


def disparity_benchmark(pop, contact, force, reference, exposure_error_factor=None):
    """Disparity benchmarks: the product identity and the exposure-offset shift
    (``Research.P2.benchmark_product``, ``benchmark_not_additive``, ``offset_shift``, ``disparity_ratio_shift``).

    Examples
    --------
    >>> d = disparity_benchmark({"A": 100, "B": 100}, {"A": 30, "B": 10}, {"A": 12, "B": 2}, reference="B")
    >>> ([round(v, 12) for v in d["resident_disparity"]], [round(v, 12) for v in d["product_check"]])
    ([6.0, 1.0], [0.0, 0.0])
    """
    g = list(pop)
    if list(contact) != g or list(force) != g:
        raise ValueError("pop, contact and force must share the same group names")
    P = np.array([float(pop[k]) for k in g])
    C = np.array([float(contact[k]) for k in g])
    F = np.array([float(force[k]) for k in g])
    if np.any(P <= 0) or np.any(C <= 0) or np.any(F <= 0):
        raise ValueError("all counts must be positive")
    if reference not in g:
        raise ValueError("reference must be one of the group names")
    r = g.index(reference)
    cd = (C / P) / (C[r] / P[r])
    fd = (F / C) / (F[r] / C[r])
    rd = (F / P) / (F[r] / P[r])
    data = {
        "group": g,
        "contact_disparity": cd,
        "force_given_contact_disparity": fd,
        "resident_disparity": rd,
        "additive_claim": cd + fd,
        "product_check": rd - cd * fd,
    }
    if exposure_error_factor is not None:
        k = np.array([float(exposure_error_factor[x]) for x in g])
        if np.any(np.isnan(k)) or np.any(k <= 0):
            raise ValueError("exposure_error_factor must be positive for every group")
        data["log_shift"] = -np.log(k)
        data["resident_disparity_corrected"] = rd * (k[r] / k)
    out = pd.DataFrame(data)
    out.attrs["theorems"] = [
        "Research.P2.benchmark_product",
        "Research.P2.benchmark_not_additive",
        "Research.P2.offset_shift",
        "Research.P2.disparity_ratio_shift",
    ]
    return out


def relative_risk_from_or(odds_ratio, base_rate=None, exposed_share=None) -> dict:
    """Relative risk from an odds ratio: the interval arrest-only data allow (``rr_between``, ``or_overstates``).

    Examples
    --------
    >>> relative_risk_from_or(3)["rr_bounds"]
    {'lower': 1, 'upper': 3.0}
    >>> round(relative_risk_from_or(3, base_rate=0.2, exposed_share=0.3)["risks"]["relative_risk"], 9)
    2.333333333
    """
    o = float(odds_ratio)
    if o != o or o <= 0:
        raise ValueError("odds_ratio must be a single positive number")
    out = {
        "rr_bounds": {"lower": min(1, o), "upper": max(1, o)},
        "theorems": ["Research.P2.or_eq_rr_mul", "Research.P2.rr_between", "Research.P2.or_overstates"],
    }
    if base_rate is not None and exposed_share is not None:
        q = float(base_rate)
        s = float(exposed_share)
        if q <= 0 or q >= 1 or s <= 0 or s >= 1:
            raise ValueError("base_rate and exposed_share must lie in (0, 1)")

        def f(b):
            return s * (o * b / (1 - b + o * b)) + (1 - s) * b - q

        b = float(brentq(f, 1e-12, 1 - 1e-12, xtol=1e-14))
        a = o * b / (1 - b + o * b)
        out["risks"] = {"exposed": a, "unexposed": b, "relative_risk": a / b}
        out["overstatement_factor"] = o / (a / b)
    return out


def deterrence_response(x, benefit, sanction, p):
    """Direction of deterrence without convexity (``certainty_monotone``, ``severity_monotone``, ``aggregate_monotone``).

    Examples
    --------
    >>> r = deterrence_response([0, 1, 2, 3], [0, 2, 3, 3.5], [0, 1, 3, 6], p=[0.1, 0.5, 1])
    >>> list(r["x_opt"])
    [3.0, 2.0, 1.0]
    """
    x = np.asarray(x, dtype=float)
    b = np.asarray(benefit, dtype=float)
    sn = np.asarray(sanction, dtype=float)
    n = x.shape[0]
    if b.shape[0] != n or sn.shape[0] != n:
        raise ValueError("x, benefit and sanction must have equal length")
    o = np.argsort(x)
    x = x[o]
    b = b[o]
    sn = sn[o]
    if np.any(np.diff(sn) <= 0):
        raise ValueError("sanction must be strictly increasing in x")
    ps = np.atleast_1d(np.asarray(p, dtype=float))
    if np.any(ps < 0):
        raise ValueError("p must be non-negative")
    rows = []
    for pp in ps:
        u = b - pp * sn
        best = u >= float(np.max(u)) - 1e-12
        rows.append({"p": float(pp), "x_opt": float(np.max(x[best])), "value": float(np.max(u))})
    out = pd.DataFrame(rows)
    out.attrs["theorems"] = [
        "Research.P8.certainty_monotone",
        "Research.P8.severity_monotone",
        "Research.P8.aggregate_monotone",
    ]
    return out


def interracial_rates(offences, population):
    """Interracial offending rates against the random-mixing null (``rate_per_offender_group``, ``dyad_ratio``).

    Examples
    --------
    >>> d = interracial_rates({"A_on_B": 120, "B_on_A": 200, "A_on_A": 900, "B_on_B": 300}, {"A": 80000, "B": 20000})
    >>> [round(v, 12) for v in d["rate_per_pair_exposure"]]
    [0.0075, 0.0125, 0.0140625, 0.075]
    """
    names = list(offences)
    parts = [k.split("_on_") for k in names]
    if any(len(p) != 2 for p in parts):
        raise ValueError("offence names must be of the form offender_on_victim")
    off = [p[0] for p in parts]
    vic = [p[1] for p in parts]
    if any(g not in population for g in off + vic):
        raise ValueError("every group in offences must appear in population")
    if any(float(v) <= 0 for v in population.values()):
        raise ValueError("populations must be positive")
    N = float(sum(float(v) for v in population.values()))
    p = {k: float(v) / N for k, v in population.items()}
    cnt = np.array([float(offences[k]) for k in names])
    pair = np.array([p[o] * p[v] * N for o, v in zip(off, vic)])
    k_hat = float(cnt.sum() / pair.sum())
    out = pd.DataFrame(
        {
            "offender": off,
            "victim": vic,
            "count": cnt,
            "rate_per_offender_group": cnt / np.array([float(population[o]) for o in off]),
            "rate_per_pair_exposure": cnt / pair,
            "null_rate_per_offender_group": np.array([k_hat * p[v] for v in vic]),
            "ratio_to_null": (cnt / pair) / k_hat,
        }
    )
    out.attrs["theorems"] = [
        "Research.P2.rate_per_offender_group",
        "Research.P2.null_slope_positive",
        "Research.P2.dyad_ratio",
        "Research.P2.pair_exposure_rate_constant",
    ]
    return out


def probability_of_necessity(p_treated, p_control) -> dict:
    """Probability of necessity bounds (``Research.P8.necessity_bounds``, ``necessity_of_monotone``).

    Examples
    --------
    >>> r = probability_of_necessity(p_treated=0.6, p_control=0.4)
    >>> ({k: round(v, 12) for k, v in r["necessary_share_bounds"].items()}, round(r["pn_monotone"], 12))
    ({'lower': 0.2, 'upper': 0.6}, 0.333333333333)
    """
    a = float(p_treated)
    b = float(p_control)
    if a < 0 or a > 1 or b < 0 or b > 1:
        raise ValueError("probabilities must be single numbers in [0, 1]")
    lo = max(0.0, a - b)
    hi = min(a, 1 - b)
    return {
        "necessary_share_bounds": {"lower": lo, "upper": hi},
        "pn_bounds": {"lower": lo / a, "upper": hi / a} if a > 0 else {"lower": float("nan"), "upper": float("nan")},
        "pn_monotone": (a - b) / a if a > 0 else float("nan"),
        "identified": math.isclose(lo, hi, rel_tol=1.5e-8, abs_tol=1.5e-8),
        "theorems": [
            "Research.P8.necessity_bounds",
            "Research.P8.necessity_of_monotone",
            "Research.P8.necessity_of_disjoint",
        ],
    }


def collider_arrest(a, e, pi_bg) -> dict:
    """Collider bias from conditioning on arrest (``collider_or_eq_background``, ``collider_or_lt_one``).

    Examples
    --------
    >>> r = collider_arrest(a=0.3, e=0.2, pi_bg=0.1)
    >>> (round(r["arrestee_or"], 12), r["population_or"])
    (0.1, 1)
    """
    a = float(a)
    e = float(e)
    pi_bg = float(pi_bg)
    if a <= 0 or a >= 1 or e <= 0 or e >= 1 or pi_bg < 0 or pi_bg > 1:
        raise ValueError("a and e must lie in (0, 1) and pi_bg in [0, 1]")
    cells = {"A1E1": a * e, "A1E0": a * (1 - e), "A0E1": (1 - a) * e, "A0E0": (1 - a) * (1 - e) * pi_bg}
    orr = (cells["A1E1"] * cells["A0E0"]) / (cells["A1E0"] * cells["A0E1"])
    tot = sum(cells.values())
    return {
        "population_or": 1,
        "arrestee_or": orr,
        "arrestee_log_or": math.log(orr) if orr > 0 else float("-inf"),
        "cells_among_arrestees": {k: v / tot for k, v in cells.items()},
        "theorems": [
            "Research.P2.population_or_one",
            "Research.P2.collider_or_eq_background",
            "Research.P2.collider_or_lt_one",
        ],
    }
