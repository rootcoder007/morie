# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P3: policing effects under interference
(``research/lean/P3Interference.lean``, ``P3HorvitzThompson.lean``, ``P3Variance.lean``, ``P3SYG.lean``).

* ``Research.P3.Model.exposure_adjustment`` / ``direct_spillover_decomposition`` / ``misspecified_exposure_bias``
* ``Research.P3.Design.ht_unbiased`` / ``ht_contrast_unbiased`` / ``ht_variance`` / ``ht_variance_estimator_unbiased``
* ``Research.P3.SYG.syg_eq_ht`` / ``syg_zero_of_const``

R parity: ``rmorie`` ``R/spillover.R``, ``src/morie_spillover.cpp``.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._rng import random_uniform

__all__ = [
    "spillover_exposure",
    "spillover_effects",
    "spillover_ht",
    "spillover_ht_variance",
    "spillover_exposure_probs",
]


def _exposure(treated, frm, to):
    n = len(treated)
    tn = [0] * n
    for a, b in zip(frm, to):
        if a < 0 or a >= n or b < 0 or b >= n:
            raise ValueError("edges must index places 1..length(treated)")
        if a == b:
            continue
        if treated[b] == 1:
            tn[a] += 1
        if treated[a] == 1:
            tn[b] += 1
    exposure = [2 if treated[i] == 1 else (1 if tn[i] > 0 else 0) for i in range(n)]
    return exposure, tn


def spillover_exposure(treated, edges):
    """Three-level exposure from a treatment on a place network (edges are 1-based pairs, as in R).

    Examples
    --------
    >>> e = spillover_exposure([1, 0, 0, 0, 1], [[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> (list(e["exposure"]), list(e["treated_neighbours"]))
    ([2, 1, 0, 1, 2], [0, 1, 0, 1, 0])
    """
    tr = [int(bool(v)) for v in treated]
    E = np.asarray(edges)
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError("edges must have two columns")
    frm = [int(v) - 1 for v in E[:, 0]]
    to = [int(v) - 1 for v in E[:, 1]]
    exposure, tn = _exposure(tr, frm, to)
    return pd.DataFrame(
        {
            "place": np.arange(1, len(tr) + 1),
            "treated": np.array(tr),
            "treated_neighbours": np.array(tn),
            "exposure": np.array(exposure),
        }
    )


def spillover_effects(y, exposure, stratum, weights=None) -> dict:
    """Direct, spillover and net effects under a stated exposure mapping (stratified exposure adjustment).

    Examples
    --------
    >>> y = [5, 4.6, 4, 6, 5.6, 5]; e = [0, 1, 2, 0, 1, 2]; s = ["a", "a", "a", "b", "b", "b"]
    >>> r = spillover_effects(y, e, s)
    >>> (round(r["spillover"], 12), round(r["direct"], 12), r["positivity"])
    (-0.4, -0.6, True)
    """
    y = np.asarray(y, dtype=float)
    ex = [int(v) for v in exposure]
    st = [str(v) for v in stratum]
    n = y.shape[0]
    if len(ex) != n or len(st) != n:
        raise ValueError("y, exposure and stratum must have equal length")
    if any(v not in (0, 1, 2) for v in ex):
        raise ValueError("exposure must take values 0, 1, 2")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    w = w / float(w.sum())
    strata = list(dict.fromkeys(st))
    cells = {}
    m_k = {}
    pooled = {}
    for s in strata:
        idx_s = [i for i in range(n) if st[i] == s]
        m_k[s] = float(sum(w[i] for i in idx_s))
        for lev in (0, 1, 2):
            idx = [i for i in idx_s if ex[i] == lev]
            cells[(s, lev)] = float(sum(w[i] * y[i] for i in idx) / sum(w[i] for i in idx)) if idx else float("nan")
        idx01 = [i for i in idx_s if ex[i] in (0, 1)]
        pooled[s] = float(sum(w[i] * y[i] for i in idx01) / sum(w[i] for i in idx01)) if idx01 else float("nan")
    positivity = not any(v != v for v in cells.values())
    means = {f"Y{lev}": sum(m_k[s] * cells[(s, lev)] for s in strata) for lev in (0, 1, 2)}
    bias_k = {s: pooled[s] - cells[(s, 0)] for s in strata}
    return {
        "means": means,
        "spillover": means["Y1"] - means["Y0"],
        "direct": means["Y2"] - means["Y1"],
        "total": means["Y2"] - means["Y0"],
        "pooling_bias": {"by_stratum": bias_k, "overall": sum(m_k[s] * bias_k[s] for s in strata)},
        "cell_means": {s: {f"Y{lev}": cells[(s, lev)] for lev in (0, 1, 2)} for s in strata},
        "stratum_weights": m_k,
        "positivity": positivity,
        "theorems": [
            "Research.P3.Model.exposure_adjustment",
            "Research.P3.Model.direct_spillover_decomposition",
            "Research.P3.Model.misspecified_exposure_bias",
        ],
    }


def spillover_ht(y, exposure, probs, joint=None) -> dict:
    """Horvitz-Thompson totals and contrasts under a randomised deployment (``ht_unbiased``, ``ht_variance``).

    Examples
    --------
    >>> pr = spillover_exposure_probs(6, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], 2, joint=True)
    >>> e = spillover_exposure([0, 1, 0, 0, 0, 1], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])["exposure"]
    >>> r = spillover_ht([5, 3.5, 4.5, 5, 4.5, 3.5], list(e), pr["marginal"], pr["joint"])
    >>> (round(r["total_effect"], 12), sorted(r.keys())[:3])
    (-0.666666666667, ['direct', 'means', 'se'])
    """
    y = np.asarray(y, dtype=float)
    ex = [int(v) for v in exposure]
    P = np.asarray(probs, dtype=float)
    n = y.shape[0]
    if len(ex) != n or P.shape[0] != n or P.shape[1] != 3:
        raise ValueError("y, exposure and probs (n x 3) must describe the same places")
    if np.any(P <= 0):
        raise ValueError("every exposure probability must be positive (positivity)")
    if any(v not in (0, 1, 2) for v in ex):
        raise ValueError("exposure must take values 0, 1, 2")
    totals = {f"T{lev}": float(sum(y[i] / P[i, lev] for i in range(n) if ex[i] == lev)) for lev in (0, 1, 2)}
    means = {k: v / n for k, v in totals.items()}
    out = {
        "totals": totals,
        "means": means,
        "spillover": means["T1"] - means["T0"],
        "direct": means["T2"] - means["T1"],
        "total_effect": means["T2"] - means["T0"],
    }
    theorems = ["Research.P3.Design.ht_unbiased", "Research.P3.Design.ht_contrast_unbiased"]
    if joint is not None:
        J = np.asarray(joint, dtype=float)
        if J.ndim != 3 or J.shape[0] != n or J.shape[1] != n or J.shape[2] != 3:
            raise ValueError("joint must be an n x n x 3 array of joint exposure probabilities")
        variance = {}
        identified = {}
        for lev in (0, 1, 2):
            pij = J[:, :, lev]
            pi_l = P[:, lev]
            ind = np.array([1.0 if ex[i] == lev else 0.0 for i in range(n)])
            oo = np.outer(ind, ind)
            if np.any(pij[oo > 0] <= 0):
                variance[f"T{lev}"] = float("nan")
            else:
                wmat = oo * (pij - np.outer(pi_l, pi_l)) / np.where(pij > 0, pij, 1.0)
                variance[f"T{lev}"] = float(np.sum(wmat * np.outer(y / pi_l, y / pi_l)))
            identified[f"T{lev}"] = bool(np.all(pij > 0))
        out["variance"] = variance
        out["se"] = {
            k: (v**0.5 if v == v and v > 0 else (0.0 if v == v else float("nan"))) for k, v in variance.items()
        }
        out["variance_identified"] = identified
        theorems += ["Research.P3.Design.ht_variance", "Research.P3.Design.ht_variance_estimator_unbiased"]
    out["theorems"] = theorems
    return out


def spillover_ht_variance(y_pot, pi, pij, form="ht"):
    """Design variance of the Horvitz-Thompson total (``ht_variance``; SYG form, ``syg_eq_ht``).

    Examples
    --------
    >>> pr = spillover_exposure_probs(6, [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], 2, joint=True)
    >>> v = spillover_ht_variance([1] * 6, pr["marginal"][:, 2], pr["joint"][:, :, 2], form="both")
    >>> (v["fixed_size"], round(v["level_count"], 12))
    (True, 2.0)
    """
    if form not in ("ht", "syg", "both"):
        raise ValueError("form must be 'ht', 'syg' or 'both'")
    yp = np.asarray(y_pot, dtype=float)
    pi = np.asarray(pi, dtype=float)
    pij = np.asarray(pij, dtype=float)
    n = yp.shape[0]
    if pi.shape[0] != n or pij.shape != (n, n):
        raise ValueError("y_pot, pi and pij must describe the same places")
    if np.any(pi <= 0):
        raise ValueError("every marginal probability must be positive")
    c = yp / pi
    ht = float(np.sum((pij - np.outer(pi, pi)) * np.outer(c, c)))
    if form == "ht":
        return ht
    diff = np.outer(c, np.ones(n)) - np.outer(np.ones(n), c)
    syg = 0.5 * float(np.sum((np.outer(pi, pi) - pij) * diff**2))
    m = float(np.sum(pi))
    fixed = float(np.max(np.abs(pij.sum(axis=1) - m * pi))) < 1e-10
    if form == "syg":
        return syg
    return {
        "ht": ht,
        "syg": syg,
        "fixed_size": fixed,
        "level_count": m,
        "theorems": [
            "Research.P3.Design.ht_variance",
            "Research.P3.SYG.syg_eq_ht",
            "Research.P3.SYG.syg_zero_of_const",
        ],
    }


def spillover_exposure_probs(n, edges, n_treated, n_draws=5000, exact_max=20000, joint=False, seed=0):
    """Exposure probabilities of a completely randomised deployment (exact enumeration, else Philox Monte Carlo).

    Examples
    --------
    >>> pr = spillover_exposure_probs(4, [[1, 2], [2, 3], [3, 4]], 1)
    >>> [round(float(v), 12) for v in pr[:, 2]]
    [0.25, 0.25, 0.25, 0.25]
    """
    n = int(n)
    n_treated = int(n_treated)
    if n_treated < 1 or n_treated >= n:
        raise ValueError("n_treated must lie in 1..n-1")
    E = np.asarray(edges)
    frm = [int(v) - 1 for v in E[:, 0]]
    to = [int(v) - 1 for v in E[:, 1]]
    counts = np.zeros((n, 3))
    jcounts = np.zeros((n, n, 3)) if joint else None

    def tally(trt):
        tr = [1 if i in trt else 0 for i in range(n)]
        e, _ = _exposure(tr, frm, to)
        for lev in (0, 1, 2):
            ind = np.array([1.0 if e[i] == lev else 0.0 for i in range(n)])
            counts[:, lev] = counts[:, lev] + ind
            if joint:
                jcounts[:, :, lev] = jcounts[:, :, lev] + np.outer(ind, ind)

    if comb(n, n_treated) <= exact_max:
        cmbs = list(combinations(range(n), n_treated))
        for cmb in cmbs:
            tally(set(cmb))
        m = len(cmbs)
    else:
        if isinstance(seed, bool) or not isinstance(seed, int | float) or seed != seed or seed < 0:
            raise ValueError("seed must be a single non-negative number")
        n_draws = int(n_draws)
        u = [float(v) for v in random_uniform(n * n_draws, seed=int(seed), stream=0)]
        for k in range(n_draws):
            block = u[k * n : (k + 1) * n]
            order = sorted(range(n), key=lambda i: block[i])
            tally(set(order[:n_treated]))
        m = n_draws
    if joint:
        return {"marginal": counts / m, "joint": jcounts / m}
    return counts / m
