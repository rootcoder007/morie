# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research ledger, the 2026-10-12 closures: eight identification results
(``research/lean/Researchproofs/P2Veil.lean``, ``P1Interaction.lean``, ``P19Error.lean``,
``P7Bound.lean``, ``P8Instrument.lean``, ``P15RIF.lean``, ``P10Moments.lean``, ``P6Panel.lean``).

* ``Research.P2.Veil.odds_ratio_identifies_bias`` / ``share_sign`` / ``exposure_shift_confounds``
* ``Research.P1.Interaction.missing_cell_identified`` / ``missing_cell_with_interaction``
* ``Research.P19Error.bhat_error`` / ``bhat_error_bound`` / ``bhat_antitone``
* ``Research.P7.Bound.gini_le_cv``
* ``Research.P8.Instrument.logcost_linear`` / ``ratio_is_discount`` / ``wald_identifies`` / ``wald_undefined``
* ``Research.P15.RIF.rif_quantile_avg`` / ``rif_quantile_avg_eq_iff`` / ``rif_gap``
* ``Research.P10.Moments.second_moments_do_not_identify``
* ``Research.P6.Panel.cross_moment_eq`` / ``one_type_cov_zero`` / ``two_type_cov``

R parity: ``rmorie`` ``R/ledger_1_4_3.R``.
"""

from __future__ import annotations

import math
import random

from morie.fn import _array_core as np
from morie.research.concentration import concentration_gini

__all__ = [
    "veil_of_darkness",
    "dark_figure_no_interaction",
    "shrinkage_factor_error",
    "concentration_null_bound",
    "deterrence_logcost",
    "wald_iv",
    "rif_quantile",
    "rif_gap",
    "contagion_moment_witness",
    "age_crime_type_cov",
    "age_crime_panel_cov",
]


def veil_of_darkness(day: dict, night: dict, reference=None, exposure_shift: dict | None = None) -> dict:
    """The veil-of-darkness odds ratio as the relative visible-race factor (``odds_ratio_identifies_bias``).

    Examples
    --------
    >>> r = veil_of_darkness({"white": 400, "black": 200}, {"white": 300, "black": 100})
    >>> round(r["bias_ratio"]["black"], 12), r["sign"]["black"]
    (1.5, 'higher')
    """
    groups = list(day)
    if set(groups) != set(night):
        raise ValueError("day and night must cover the same groups")
    reference = groups[0] if reference is None else reference
    if reference not in day:
        raise ValueError("reference is not a group")
    d = {g: float(day[g]) for g in groups}
    n = {g: float(night[g]) for g in groups}
    if any(v < 0 for v in d.values()) or any(v <= 0 for v in n.values()):
        raise ValueError("counts must be non-negative, darkness counts positive")
    sd, sn = sum(d.values()), sum(n.values())
    share_day = {g: d[g] / sd for g in groups}
    share_night = {g: n[g] / sn for g in groups}
    ratio = {g: (d[g] / d[reference]) / (n[g] / n[reference]) for g in groups}
    sign = {}
    for g in groups:
        diff = share_day[g] - share_night[g]
        sign[g] = "equal" if abs(diff) < 1e-12 else ("higher" if diff > 0 else "lower")
    out = {
        "share_day": share_day,
        "share_night": share_night,
        "bias_ratio": ratio,
        "sign": sign,
        "theorem": "Research.P2.Veil.odds_ratio_identifies_bias",
    }
    if exposure_shift is not None:
        if any(g not in exposure_shift for g in groups):
            raise ValueError("exposure_shift must be named over the groups")
        out["bias_ratio_adjusted"] = {
            g: ratio[g] / (float(exposure_shift[g]) / float(exposure_shift[reference])) for g in groups
        }
    return out


def dark_figure_no_interaction(cells: dict, interaction=None) -> dict:
    """The never-listed cell under no three-way interaction (``missing_cell_identified``).

    Examples
    --------
    >>> r = dark_figure_no_interaction({"n100": 120, "n010": 80, "n001": 60, "n110": 30, "n101": 24, "n011": 16, "n111": 8})
    >>> round(r["missing"], 12), round(r["total"], 12)
    (400.0, 738.0)
    """
    need = ["n100", "n010", "n001", "n110", "n101", "n011", "n111"]
    if any(k not in cells for k in need):
        raise ValueError("cells must be named " + ", ".join(need))
    n = {k: float(cells[k]) for k in need}
    if any(v < 0 for v in n.values()):
        raise ValueError("counts must be non-negative")
    if n["n110"] == 0 or n["n101"] == 0 or n["n011"] == 0:
        raise ValueError("the three pairwise cells must be positive")
    missing = n["n100"] * n["n010"] * n["n001"] * n["n111"] / (n["n110"] * n["n101"] * n["n011"])
    observed = sum(n.values())
    out = {
        "missing": missing,
        "observed": observed,
        "total": observed + missing,
        "theorem": "Research.P1.Interaction.missing_cell_identified",
    }
    if interaction is not None:
        t = [float(v) for v in interaction]
        out["sensitivity"] = [
            {"interaction": v, "missing": missing * math.exp(-v), "total": observed + missing * math.exp(-v)} for v in t
        ]
    return out


def _wvar(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    m = float(np.sum(w * v) / np.sum(w))
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


def shrinkage_factor_error(y, noise_variance: float, weights=None, total_variance=None) -> dict:
    """Error of the plug-in shrinkage factor when the total mass is estimated (``bhat_error_bound``).

    Examples
    --------
    >>> r = shrinkage_factor_error([3, 5, 9, 2, 7, 4], noise_variance=5, total_variance=6.0)
    >>> round(r["B_hat"], 12), round(r["error"], 12)
    (0.882352941176, 0.049019607843)
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 3 or np.any(np.isnan(y)):
        raise ValueError("y needs at least three non-missing counts")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    if noise_variance < 0:
        raise ValueError("noise_variance must be a single non-negative number")
    s_hat = _wvar(y, w)
    if s_hat <= 0:
        raise ValueError("the counts have no spread; the total mass is zero")
    loo = np.array([_wvar(np.delete(y, i), np.delete(w, i)) for i in range(n)])
    s_se = math.sqrt((n - 1) / n * float(np.sum((loo - loo.mean()) ** 2)))
    b_hat = noise_variance / s_hat
    lo = max(s_hat - 2 * s_se, 2.220446049250313e-16)
    hi = s_hat + 2 * s_se
    out = {
        "B_hat": b_hat,
        "S_y_hat": s_hat,
        "S_y_se": s_se,
        "B_interval": (noise_variance / hi, noise_variance / lo),
        "error_bound": b_hat * (2 * s_se) / s_hat,
        "theorem": "Research.P19Error.bhat_error_bound",
    }
    if total_variance is not None:
        b_star = noise_variance / float(total_variance)
        out["B_star"] = b_star
        out["error"] = b_hat - b_star
        out["bound_holds"] = abs(out["error"]) <= b_star * abs(s_hat - total_variance) / s_hat + 1e-12
    return out


def concentration_null_bound(x, n_sim: int = 2000, seed: int = 1) -> dict:
    """The Gini against its coefficient-of-variation bound and a simulated Poisson null (``gini_le_cv``).

    Examples
    --------
    >>> r = concentration_null_bound([0, 0, 1, 2, 9, 0, 3], n_sim=200)
    >>> round(r["gini"], 12), round(r["cv_bound"], 12), r["gini"] <= r["cv_bound"]
    (0.666666666667, 0.988826464946, True)
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2 or np.any(np.isnan(x)) or np.any(x < 0) or float(x.sum()) <= 0:
        raise ValueError("x must be non-negative counts with a positive total")
    m = float(x.mean())
    v = float(np.mean((x - m) ** 2))
    g = concentration_gini(x)
    rng = random.Random(seed)
    sims = []
    for _ in range(n_sim):
        s = [_rpois(rng, m) for _ in range(n)]
        sims.append(0.0 if sum(s) == 0 else concentration_gini(s))
    sims = np.asarray(sims, dtype=float)
    return {
        "gini": g,
        "mean": m,
        "variance": v,
        "cv_bound": math.sqrt(v / 2) / m,
        "null_bound": 1 / math.sqrt(2 * m),
        "null_mean": float(sims.mean()),
        "null_q95": float(np.quantile(sims, 0.95)),
        "excess": g - float(sims.mean()),
        "p_value": float(np.mean(sims >= g)),
        "n": n,
        "theorem": "Research.P7.Bound.gini_le_cv",
    }


def _rpois(rng, mu):
    # Knuth's method for a Poisson draw from a seeded generator
    L = math.exp(-mu)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def deterrence_logcost(p, s, c, y=None, rho=None) -> dict:
    """The expected-cost response on the log design and the discount it implies (``logcost_linear``).

    Examples
    --------
    >>> p, s, c = [0.2, 0.5, 0.8, 0.4], [2, 5, 3, 8], [1, 4, 2, 6]
    >>> y = [2 - 0.8 * (math.log(pi) + math.log(si) - 0.1 * ci) for pi, si, ci in zip(p, s, c)]
    >>> round(deterrence_logcost(p, s, c, y)["implied_discount"], 10)
    0.1
    """
    p = np.asarray(p, dtype=float)
    s = np.asarray(s, dtype=float)
    c = np.asarray(c, dtype=float)
    if np.any(p <= 0) or np.any(s <= 0):
        raise ValueError("p and s must be positive")
    n = p.shape[0]
    if s.shape[0] != n or c.shape[0] != n:
        raise ValueError("p, s and c must have one value per observation")
    X = np.column_stack([np.ones(n), np.log(p), np.log(s), c])
    rank = int(np.linalg.matrix_rank(X))
    out = {"rank": rank, "identified": rank == 4, "theorem": "Research.P8.Instrument.logcost_linear"}
    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.shape[0] != n:
            raise ValueError("y must have one value per observation")
        if rank < 4:
            raise ValueError("the log design is rank deficient: a dimension held fixed is not identified")
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        out["coefficients"] = {"(Intercept)": float(b[0]), "log_p": float(b[1]), "log_s": float(b[2]), "c": float(b[3])}
        out["implied_discount"] = -float(b[3]) / float(b[1]) if b[1] != 0 else float("nan")
    if rho is not None:
        out["expected_cost"] = (p * s * np.exp(-rho * c)).tolist()
        out["log_expected_cost"] = (np.log(p) + np.log(s) - rho * c).tolist()
    return out


def wald_iv(y, p, z) -> dict:
    """The Wald instrumental-variable ratio (``wald_identifies``, ``wald_undefined``).

    Examples
    --------
    >>> wald_iv([1, 3, 2, 4], [0, 1, 0, 1], [0, 1, 0, 1])["beta"]
    2.0
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    z = np.asarray(z, dtype=float)
    n = y.shape[0]
    if p.shape[0] != n or z.shape[0] != n or n < 2:
        raise ValueError("y, p and z must have the same length, at least 2")

    def cov(a, b):
        return float(np.mean(a * b) - np.mean(a) * np.mean(b))

    czy, czp = cov(z, y), cov(z, p)
    if abs(czp) < 1e-12:
        return {
            "beta": float("nan"),
            "cov_zy": czy,
            "cov_zp": czp,
            "relevant": False,
            "theorem": "Research.P8.Instrument.wald_undefined",
        }
    beta = czy / czp
    return {
        "beta": beta,
        "cov_zy": czy,
        "cov_zp": czp,
        "relevant": True,
        "alpha": float(np.mean(y) - beta * np.mean(p)),
        "theorem": "Research.P8.Instrument.wald_identifies",
    }


def rif_quantile(y, tau: float, weights=None, q=None, f=None) -> dict:
    """Recentered influence function of a quantile (``rif_quantile_avg``).

    Examples
    --------
    >>> r = rif_quantile([1, 2, 3, 4, 5, 6, 7, 8], 0.5, q=4, f=0.125)
    >>> round(r["mean_rif"], 12), round(r["recentering"], 12)
    (4.0, 0.0)
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2 or np.any(np.isnan(y)):
        raise ValueError("y needs at least two non-missing values")
    if not 0 < tau < 1:
        raise ValueError("tau must lie in (0, 1)")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    if q is None:
        o = np.argsort(y, kind="stable")
        cw = np.cumsum(w[o]) / float(w.sum())
        q = float(y[o][int(np.argmax(cw >= tau))])
    if f is None:
        f = _kde_at(y, w, q)
    if f <= 0:
        raise ValueError("the density at q must be positive")
    below = (y <= q).astype(float)
    rif = q + (tau - below) / f
    F_q = float(np.sum(w * below) / np.sum(w))
    return {
        "rif": rif.tolist(),
        "q": float(q),
        "f": float(f),
        "F_q": F_q,
        "mean_rif": float(np.sum(w * rif) / np.sum(w)),
        "recentering": (tau - F_q) / f,
        "theorem": "Research.P15.RIF.rif_quantile_avg",
    }


def _kde_at(y, w, q):
    # Gaussian kernel with Silverman's rule (R's bw.nrd0), weights normalised
    n = y.shape[0]
    sd = float(np.std(y, ddof=1))
    qs = np.quantile(y, [0.25, 0.75])
    iqr = float(qs[1] - qs[0]) / 1.34
    lo = min(sd, iqr) if iqr > 0 else sd
    if lo <= 0:
        lo = abs(float(y[0])) if y[0] != 0 else 1.0
    bw = 0.9 * lo * n ** (-0.2)
    ww = w / float(w.sum())
    return float(np.sum(ww * np.exp(-0.5 * ((q - y) / bw) ** 2) / (bw * math.sqrt(2 * math.pi))))


def rif_gap(yA, yB, tau: float, wA=None, wB=None) -> dict:
    """The gap of two groups' RIF means as quantile gap plus recentering (``rif_gap``)."""
    a = rif_quantile(yA, tau, wA)
    b = rif_quantile(yB, tau, wB)
    return {
        "A": a,
        "B": b,
        "gap": a["mean_rif"] - b["mean_rif"],
        "quantile_gap": a["q"] - b["q"],
        "recentering_gap": a["recentering"] - b["recentering"],
        "theorem": "Research.P15.RIF.rif_gap",
    }


def contagion_moment_witness(statistic=None) -> dict:
    """Two count laws with equal mean and variance but different zero mass (``second_moments_do_not_identify``).

    Examples
    --------
    >>> w = contagion_moment_witness(lambda m, v: v / m)
    >>> w["dispersion"], w["zero_probability"]
    ((1.0, 1.0), (0.5, 0.3333333333333333))
    """
    p = [1 / 2, 0.0, 1 / 2, 0.0]
    q = [1 / 3, 1 / 2, 0.0, 1 / 6]
    k = [0, 1, 2, 3]

    def mom(r, j):
        return sum(ri * ki**j for ri, ki in zip(r, k))

    mean = (mom(p, 1), mom(q, 1))
    var = (mom(p, 2) - mom(p, 1) ** 2, mom(q, 2) - mom(q, 1) ** 2)
    out = {
        "p": p,
        "q": q,
        "support": k,
        "mean": mean,
        "variance": var,
        "dispersion": (var[0] / mean[0], var[1] / mean[1]),
        "zero_probability": (p[0], q[0]),
        "theorem": "Research.P10.Moments.second_moments_do_not_identify",
    }
    if statistic is not None:
        out["statistic"] = (statistic(mean[0], var[0]), statistic(mean[1], var[1]))
    return out


def age_crime_type_cov(w, rate_a, rate_b) -> dict:
    """Cross-age covariance of a type mixture (``cross_moment_eq``, ``two_type_cov``).

    Examples
    --------
    >>> age_crime_type_cov([0.5, 0.5], [1, 3], [3, 1])["cross_cov"]
    -1.0
    """
    w = np.asarray(w, dtype=float)
    ra = np.asarray(rate_a, dtype=float)
    rb = np.asarray(rate_b, dtype=float)
    if w.shape[0] != ra.shape[0] or w.shape[0] != rb.shape[0]:
        raise ValueError("one weight and two rates per type")
    if abs(float(w.sum()) - 1) > 1e-10 or np.any(w < 0):
        raise ValueError("w must be non-negative weights summing to one")
    ma, mb = float(np.sum(w * ra)), float(np.sum(w * rb))
    cc = float(np.sum(w * (ra - ma) * (rb - mb)))
    return {
        "aggregate": (ma, mb),
        "cross_cov": cc,
        "one_type": abs(cc) < 1e-12,
        "theorem": "Research.P6.Panel.cross_moment_eq",
    }


def age_crime_panel_cov(y_a, y_b, n_perm: int = 2000, seed: int = 1) -> dict:
    """Sample cross-age covariance with an exact one-type permutation null (``one_type_cov_zero``)."""
    ya = np.asarray(y_a, dtype=float)
    yb = np.asarray(y_b, dtype=float)
    n = ya.shape[0]
    if yb.shape[0] != n or n < 3 or np.any(np.isnan(ya)) or np.any(np.isnan(yb)):
        raise ValueError("y_a and y_b need the same individuals, at least three")

    def cov(a, b):
        return float(np.mean(a * b) - np.mean(a) * np.mean(b))

    obs = cov(ya, yb)
    rng = random.Random(seed)
    idx = list(range(n))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        if abs(cov(ya, yb[idx])) >= abs(obs) - 1e-12:
            count += 1
    return {
        "cross_cov": obs,
        "p_value": (1 + count) / (n_perm + 1),
        "n": n,
        "theorem": "Research.P6.Panel.one_type_cov_zero",
    }


def feedback_loop_envelope(r0: float, b0: float, a: float = 1.0, n: int = 100, t: float = 0.25) -> dict:
    """Drift envelope of the discovery urn: variance at most a/(4(r0+b0)) (``Research.P4.Envelope.var_le``).

    Examples
    --------
    >>> e = feedback_loop_envelope(1, 1, 1, n=40)
    >>> (e["variance_bound"], round(e["variance_n"], 12), round(e["mean_n"], 12))
    (0.125, 0.079365079365, 0.5)
    """
    if not (r0 > 0 and b0 > 0 and a > 0):
        raise ValueError("r0, b0 and a must be positive numbers")
    if not t > 0:
        raise ValueError("t must be a positive half-width")
    n = int(n)
    if n < 0:
        raise ValueError("n must be a non-negative horizon")

    def share(k, j):
        return (r0 + j * a) / (r0 + b0 + k * a)

    q = [1.0]
    for k in range(1, n + 1):
        nq = [0.0] * (k + 1)
        for j in range(k):
            s = share(k - 1, j)
            nq[j] += q[j] * (1 - s)
            nq[j + 1] += q[j] * s
        q = nq
    s_n = [share(n, j) for j in range(n + 1)]
    share0 = r0 / (r0 + b0)
    mean_n = sum(qj * sj for qj, sj in zip(q, s_n))
    var_n = sum(qj * (sj - share0) ** 2 for qj, sj in zip(q, s_n))
    exit_n = sum(qj for qj, sj in zip(q, s_n) if abs(sj - share0) >= t)
    return {
        "share0": share0,
        "variance_bound": a / (4 * (r0 + b0)),
        "chebyshev_bound": a / (4 * (r0 + b0) * t**2),
        "variance_n": var_n,
        "mean_n": mean_n,
        "exit_n": exit_n,
        "law": q,
        "n": n,
        "theorem": "Research.P4.Envelope.chebyshev",
    }


def spillover_ring_bound(outer_share, gamma_max: float, gamma=None) -> dict:
    """Ring mis-specification bound: |E gap| <= gamma_max * mean(outer_share) (``Research.P3.Ring.expected_gap_le``).

    Examples
    --------
    >>> r = spillover_ring_bound([0.1, 0.3, 0.2], gamma_max=2)
    >>> (round(r["mean_share"], 12), round(r["bound"], 12))
    (0.2, 0.4)
    """
    s = [float(v) for v in outer_share]
    if not s or any(v != v or v < 0 or v > 1 for v in s):
        raise ValueError("outer_share must be probabilities in [0, 1]")
    if not gamma_max >= 0:
        raise ValueError("gamma_max must be a non-negative bound")
    gap = None
    if gamma is not None:
        g = [float(v) for v in gamma]
        if len(g) != len(s):
            raise ValueError("gamma must have one value per unit")
        if any(abs(v) > gamma_max + 1e-12 for v in g):
            raise ValueError("gamma exceeds gamma_max somewhere")
        gap = sum(a * b for a, b in zip(g, s)) / len(s)
    mean_share = sum(s) / len(s)
    return {
        "mean_share": mean_share,
        "bound": gamma_max * mean_share,
        "expected_gap": gap,
        "theorems": ("Research.P3.Ring.expected_gap_eq", "Research.P3.Ring.expected_gap_le"),
    }


def rank_persistence_null(n: int, k: int, observed=None) -> dict:
    """Hypergeometric null for the overlap of two top-k sets of n places (``Research.P7.Persistence.overlap_prob``).

    Examples
    --------
    >>> r = rank_persistence_null(158, 16, observed=9)
    >>> (round(r["mean"], 12), r["p_value"] < 1e-4, round(sum(r["probability"]), 12))
    (1.620253164557, True, 1.0)
    """
    from math import comb

    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")
    prob = [comb(k, j) * comb(n - k, k - j) / comb(n, k) for j in range(k + 1)]
    p_value = None
    if observed is not None:
        observed = int(observed)
        if observed < 0 or observed > k:
            raise ValueError("observed must be an overlap between 0 and k")
        p_value = sum(prob[observed:])
    return {
        "overlap": list(range(k + 1)),
        "probability": prob,
        "mean": k * k / n,
        "p_value": p_value,
        "theorems": (
            "Research.P7.Persistence.card_overlapSets",
            "Research.P7.Persistence.overlap_prob",
            "Research.P7.Persistence.overlap_prob_sum",
        ),
    }


def hotspot_threshold(
    A0: float, omega: float, eps: float, D: float, gamma: float, eta: float, z: float = 1.0, k=None
) -> dict:
    """Hot-spot threshold of the Short-Brantingham model (``Research.P4.Hotspot``).

    The uniform steady state is B = eps D gamma / omega, A = A0 + B, rho = gamma / A
    (``steady_iff``); the mode-k determinant is P k^2 - Q k + R with P = eta D^2 / z,
    Q = D (3 eps D rho - omega - eta A / z), R = omega A, and the trace is negative, so a mode
    grows iff the determinant is negative (``unstable_iff``). Q <= 0 means no hot spots at any
    wavelength (``no_hotspots_below_threshold``); Q > 0 with Q^2 > 4 P R makes the critical
    mode k* = Q / (2 P) grow (``hotspots_above_threshold``).

    Examples
    --------
    >>> hotspot_threshold(A0=1, omega=1, eps=2, D=1, gamma=1, eta=0.01)["unstable"]
    True
    >>> hotspot_threshold(A0=1, omega=1, eps=0, D=1, gamma=1, eta=0.01)["unstable"]
    False
    """
    import math

    def one(x, name, lower, strict=True):
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a single finite number") from None
        if not math.isfinite(v) or (v <= lower if strict else v < lower):
            raise ValueError(f"{name} must be a single finite number {'above' if strict else 'at least'} {lower}")
        return v

    A0 = one(A0, "A0", 0)
    omega = one(omega, "omega", 0)
    eps = one(eps, "eps", 0, strict=False)
    D = one(D, "D", 0)
    gamma = one(gamma, "gamma", 0)
    eta = one(eta, "eta", 0, strict=False)
    z = one(z, "z", 0)
    Bbar = eps * D * gamma / omega
    Abar = A0 + Bbar
    rhobar = gamma / Abar
    a = -omega + eps * D * rhobar
    b = eps * D * Abar
    c0 = -rhobar
    c1 = 2 * D * rhobar / Abar
    d = -Abar
    d1 = eta * D / z
    d2 = D
    P = d1 * d2
    Q = a * d2 + d * d1 + b * c1
    R = a * d - b * c0

    def tr_k(kk):
        return (a - d1 * kk) + (d - d2 * kk)

    def det_k(kk):
        return (a - d1 * kk) * (d - d2 * kk) - b * (c0 + c1 * kk)

    def growth_k(kk):
        t = tr_k(kk)
        dd = det_k(kk)
        return (t + math.sqrt(max(t * t - 4 * dd, 0.0))) / 2

    lhs = 3 * eps * D * rhobar
    rhs = omega + eta * Abar / z
    if P > 0:
        unstable = Q > 0 and Q * Q > 4 * P * R
        k_crit = Q / (2 * P) if Q > 0 else None
    else:
        unstable = Q > 0
        k_crit = 2 * R / Q if Q > 0 else None
    det_crit = None if k_crit is None else det_k(k_crit)
    growth_crit = None if k_crit is None else growth_k(k_crit)
    dispersion = None
    if k is not None:
        ks = [float(v) for v in k]
        if any(v != v or v < 0 for v in ks):
            raise ValueError("k must be non-negative wavenumbers-squared")
        dispersion = [{"k": v, "trace": tr_k(v), "det": det_k(v), "growth": growth_k(v)} for v in ks]
    return {
        "steady": {"B": Bbar, "A": Abar, "rho": rhobar},
        "P": P,
        "Q": Q,
        "R": R,
        "threshold": {"reinforcement": lhs, "decay_plus_diffusion": rhs},
        "unstable": unstable,
        "k_crit": k_crit,
        "det_crit": det_crit,
        "growth_crit": growth_crit,
        "dispersion": dispersion,
        "theorems": (
            "Research.P4.Hotspot.steady_iff",
            "Research.P4.Hotspot.unstable_iff",
            "Research.P4.Hotspot.no_hotspots_below_threshold",
            "Research.P4.Hotspot.hotspots_above_threshold",
        ),
    }


def dark_figure_comparable(VA: float, RA: float, RB: float, lam: float = 1.0) -> dict:
    """The dark figure from a survey covering only the offence set A (``Research.P1.Comparable``).

    The survey identifies VA and so r_A = RA / VA (``rate_on_A``); the overall rate is at most
    (RA + RB) / (VA + RB) (``total_rate_le``) and has no positive lower bound without an
    assumption on B. Under r_B >= lam r_A the total count lies in [VA + RB, VA + RB/(lam r_A)]
    (``total_interval``) and the dark figure in [VA - RA, VA - RA + RB (1/(lam r_A) - 1)]
    (``dark_interval``), an interval of width RB (1 - lam r_A)/(lam r_A) (``dark_figure_width``).

    Examples
    --------
    >>> r = dark_figure_comparable(VA=1000, RA=400, RB=300, lam=0.5)
    >>> (r["rate_A"], r["total"], r["dark_figure"], r["width"])
    (0.4, (1300.0, 2500.0), (600.0, 1800.0), 1200.0)
    """
    import math

    def num(x, name):
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a single finite number") from None
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be a single finite number at least 0")
        return v

    VA, RA, RB, lam = num(VA, "VA"), num(RA, "RA"), num(RB, "RB"), num(lam, "lam")
    if VA <= 0:
        raise ValueError("VA must be positive: the survey must see some comparable offences")
    if RA > VA:
        raise ValueError("RA exceeds VA: a record is an offence, so the recorded count cannot exceed the true one")
    if lam <= 0 or lam > 1:
        raise ValueError("lam must be in (0, 1]")
    rate_A = RA / VA
    if rate_A <= 0:
        raise ValueError("RA must be positive for the bound: with nothing recorded on A, lam r_A = 0")
    k = lam * rate_A
    return {
        "rate_A": rate_A,
        "rate_upper": (RA + RB) / (VA + RB),
        "total": (VA + RB, VA + RB / k),
        "dark_figure": (VA - RA, VA - RA + RB * (1 / k - 1)),
        "width": RB * (1 - k) / k,
        "theorems": (
            "Research.P1.Comparable.rate_on_A",
            "Research.P1.Comparable.total_rate_le",
            "Research.P1.Comparable.total_interval",
            "Research.P1.Comparable.dark_interval",
            "Research.P1.Comparable.dark_figure_width",
        ),
    }
