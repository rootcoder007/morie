# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research ledger, the library-scan closures of 2026-10-12: the Python side of twelve results.

Every statement is a machine-checked theorem (Lean 4 + Mathlib, 0 sorry, standard axioms
only) in ``research/lean/Researchproofs/`` of the R package (rmorie); the function names
mirror the R functions ``morie_*`` without the prefix.

=========================  ============================================================
Lean file                  theorems
=========================  ============================================================
P5Attenuation.lean         Research.P5.Attenuation.two_regressor_coeffs (scan A4)
P4Friedman.lean            Research.P4.Friedman.meanDiff_succ (scan C2)
P4Ruin.lean                Research.P4.Ruin.ruin_harmonic (scan C4)
P7Cox.lean                 Research.P7.Cox.total_variance, llr_zero (scan C12, B24)
P1Nonresponse.lean         Research.P1.Nonresponse.region_width (scan B10)
P8IVInequality.lean        Research.P8.IVInequality.instrumental_inequality (scan B12)
P2Misclassification.lean   Research.P2.Misclassification.binary_attenuation, ordinal_witness (B14)
P11Amplification.lean      Research.P11.Amplification.level_curve (scan B18)
P13Design.lean             Research.P13.Design.quad_form (scan A10, B9)
P3Basu.lean                Research.P3.Basu.ht_variance (scan B8)
P4Contraction.lean         Research.P4.Contraction.decay (scan C7)
P8Inspection.lean          Research.P8.Inspection.conserved (scan A18, C24)
=========================  ============================================================
"""

from __future__ import annotations

import math
import statistics

__all__ = [
    "attenuation_two_regressor",
    "friedman_urn",
    "ruin_probability",
    "cox_overdispersion",
    "zip_vuong_boundary",
    "nonresponse_region",
    "instrumental_inequality",
    "misclassified_rd",
    "misclassified_contrast",
    "sensitivity_amplification",
    "design_effect_direction",
    "ht_elephant",
    "attractiveness_contraction",
    "inspection_game",
]


def attenuation_two_regressor(
    beta1: float, beta2: float, s11: float, s22: float, s12: float = 0.0, su2: float = 0.0
) -> dict:
    """Measurement error attenuates only in the one-regressor case.

    The one-regressor slope on the observed regressor is ``beta1 * s11 / (s11 + su2)``; with
    an error-free second regressor the population coefficients are
    ``b1 = beta1 (s11 s22 - s12^2) / D*`` and ``b2 = beta2 + beta1 su2 s12 / D*`` with
    ``D* = (s11 + su2) s22 - s12^2`` (Lean ``Research.P5.Attenuation.two_regressor_coeffs``):
    the error-free coefficient is biased unless ``beta1``, ``su2`` or ``s12`` is zero, and the
    bias has the sign of ``beta1 * s12``.

    >>> r = attenuation_two_regressor(1, 0.5, 1, 1, 0.6, 0.5)
    >>> round(r["b2"] - 0.5, 6)
    0.263158
    """
    if s11 <= 0 or s22 <= 0:
        raise ValueError("s11 and s22 must be positive variances")
    if su2 < 0:
        raise ValueError("su2 must be non-negative")
    if s12 * s12 >= s11 * s22:
        raise ValueError("s12 must satisfy s12^2 < s11 * s22")
    dstar = (s11 + su2) * s22 - s12 * s12
    b1 = beta1 * (s11 * s22 - s12 * s12) / dstar
    b2 = beta2 + beta1 * su2 * s12 / dstar
    bias = b2 - beta2
    return {
        "slope_single": beta1 * s11 / (s11 + su2),
        "reliability": s11 / (s11 + su2),
        "b1": b1,
        "b2": b2,
        "bias_b2": bias,
        "bias_b2_sign": "positive" if bias > 0 else "negative" if bias < 0 else "zero",
        "dstar": dstar,
        "theorem": "Research.P5.Attenuation.two_regressor_coeffs",
    }


def friedman_urn(b0: float, r0: float, n: int = 100) -> dict:
    """The Friedman urn: cross-colour reinforcement pulls the share to one half.

    The expected difference after ``n`` steps is ``D0 (T0 - 1) / (T0 + n - 1)``, which solves
    ``E[D_{n+1}] = E[D_n] (1 - 1/T_n)`` (Lean ``Research.P4.Friedman.meanDiff_succ``) and tends
    to zero; the mean share ``1/2 + E[D_n] / (2 T_n)`` tends to one half.

    >>> friedman_urn(9, 1, 3)["mean_diff"]
    [8.0, 7.2, 6.545454545454546, 6.0]
    """
    if b0 < 0 or r0 < 0 or b0 + r0 <= 1:
        raise ValueError("b0 + r0 must exceed 1")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    T0 = b0 + r0
    D0 = b0 - r0
    step = list(range(n + 1))
    mean_diff = [D0 * (T0 - 1) / (T0 + k - 1) for k in step]
    return {
        "step": step,
        "total": [T0 + k for k in step],
        "mean_diff": mean_diff,
        "mean_share": [0.5 + d / (2 * (T0 + k)) for k, d in zip(step, mean_diff)],
        "theorem": "Research.P4.Friedman.meanDiff_succ",
    }


def ruin_probability(p: float, N: int, k=None) -> dict:
    """Absorption probability of a finite-capacity allocation.

    With ``rho = (1 - p) / p`` the probability of ending at zero from ``k`` is
    ``(rho^k - rho^N) / (1 - rho^N)``, the solution of ``p_k = p p_{k+1} + (1 - p) p_{k-1}``
    with ``p_0 = 1``, ``p_N = 0`` (Lean ``Research.P4.Ruin.ruin_harmonic``); at ``p = 1/2`` it
    is ``1 - k/N``.

    >>> ruin_probability(0.5, 4)["prob"]
    [1.0, 0.75, 0.5, 0.25, 0.0]
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    N = int(N)
    ks = list(range(N + 1)) if k is None else [int(x) for x in (k if isinstance(k, list | tuple) else [k])]
    if N < 1 or any(x < 0 or x > N for x in ks):
        raise ValueError("k must lie in 0..N")
    rho = (1 - p) / p
    symmetric = abs(p - 0.5) < 1e-12
    prob = [1 - x / N for x in ks] if symmetric else [(rho**x - rho**N) / (1 - rho**N) for x in ks]
    return {"k": ks, "prob": prob, "rho": rho, "theorem": "Research.P4.Ruin.ruin_harmonic"}


def cox_overdispersion(w, lam) -> dict:
    """Any intensity heterogeneity forces overdispersion.

    A count whose intensity is a finite mixture has ``Var N = E N + Var Lambda`` (Lean
    ``Research.P7.Cox.total_variance``), so the variance is at least the mean, with equality
    iff the intensity is constant on the support.

    >>> r = cox_overdispersion([0.7, 0.3], [1, 5])
    >>> round(r["variance"] - r["mean"], 6) == round(r["intensity_var"], 6)
    True
    """
    w = [float(x) for x in w]
    lam = [float(x) for x in lam]
    if len(w) != len(lam) or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("w must be non-negative weights, one per intensity")
    if any(x < 0 for x in lam):
        raise ValueError("lam must be non-negative")
    s = sum(w)
    w = [x / s for x in w]
    m = sum(a * b for a, b in zip(w, lam))
    iv = sum(a * (b - m) ** 2 for a, b in zip(w, lam))
    v = sum(a * (b + b * b) for a, b in zip(w, lam)) - m * m
    return {
        "mean": m,
        "variance": v,
        "intensity_var": iv,
        "dispersion": v / m if m > 0 else float("nan"),
        "constant_intensity": all(abs(b - m) < 1e-12 for a, b in zip(w, lam) if a > 0),
        "theorem": "Research.P7.Cox.total_variance",
    }


def zip_vuong_boundary(counts, lam: float | None = None) -> dict:
    """Vuong's test of zero-inflated against plain Poisson at the boundary.

    At ``pi = 0`` the two pmfs agree at every count (Lean
    ``Research.P7.Cox.zip_zero_eq_poisson``), so every pointwise log-likelihood ratio is zero
    (``llr_zero``), their standard deviation is zero and the statistic is 0/0.

    >>> zip_vuong_boundary([0, 0, 1, 2])["sd"]
    0.0
    """
    counts = [int(c) if float(c) == int(c) else None for c in counts]
    if any(c is None or c < 0 for c in counts):
        raise ValueError("counts must be non-negative integers")
    if lam is None:
        lam = sum(counts) / len(counts)

    def logpois(n):
        return -lam + n * math.log(lam) - math.lgamma(n + 1) if lam > 0 else (0.0 if n == 0 else -math.inf)

    llr = [math.log(0 + (1 - 0) * math.exp(logpois(n))) - logpois(n) for n in counts]
    sd = statistics.stdev(llr) if len(llr) > 1 else 0.0
    return {"llr": llr, "sd": sd, "statistic": float("nan"), "theorem": "Research.P7.Cox.llr_zero"}


def nonresponse_region(p_obs, r) -> dict:
    """The identification region of a proportion under nonresponse.

    With reporting rate ``r`` and observed share ``p_obs`` among reporters the proportion lies
    in ``[r p_obs, r p_obs + 1 - r]``, both ends attained, of width exactly ``1 - r`` (Lean
    ``Research.P1.Nonresponse.region_width``).

    >>> nonresponse_region(0.4, [0.106, 0.5])["width"]
    [0.894, 0.5]
    """
    ps = [float(x) for x in (p_obs if isinstance(p_obs, list | tuple) else [p_obs])]
    rs = [float(x) for x in (r if isinstance(r, list | tuple) else [r])]
    if any(x < 0 or x > 1 for x in ps + rs):
        raise ValueError("p_obs and r must lie in [0, 1]")
    n = max(len(ps), len(rs))
    ps = [ps[i % len(ps)] for i in range(n)]
    rs = [rs[i % len(rs)] for i in range(n)]
    return {
        "p_obs": ps,
        "r": rs,
        "lower": [a * b for a, b in zip(rs, ps)],
        "upper": [a * b + 1 - a for a, b in zip(rs, ps)],
        "width": [1 - a for a in rs],
        "theorem": "Research.P1.Nonresponse.region_width",
    }


def instrumental_inequality(p, tol: float = 1e-8) -> dict:
    """Pearl's instrumental inequality on a finite table.

    ``p[y][x][z]`` is ``P(y, x | z)`` with every z-slice summing to one; the model implies
    ``sum_y max_z P(y, x | z) <= 1`` for every ``x`` (Lean
    ``Research.P8.IVInequality.instrumental_inequality``).

    >>> p = [[[0.4, 0.2], [0.3, 0.1]], [[0.1, 0.3], [0.2, 0.4]]]
    >>> instrumental_inequality(p)["passes"]
    True
    """
    ny = len(p)
    nx = len(p[0])
    nz = len(p[0][0])
    if any(len(row) != nx for row in p) or any(len(c) != nz for row in p for c in row):
        raise ValueError("p must be a (y, x, z) array")
    if any(p[y][x][z] < 0 for y in range(ny) for x in range(nx) for z in range(nz)):
        raise ValueError("p must be a non-negative (y, x, z) array")
    for z in range(nz):
        if abs(sum(p[y][x][z] for y in range(ny) for x in range(nx)) - 1) > tol:
            raise ValueError("every z-slice of p must sum to one")
    per_x = [sum(max(p[y][x][z] for z in range(nz)) for y in range(ny)) for x in range(nx)]
    return {
        "per_x": per_x,
        "max": max(per_x),
        "passes": all(v <= 1 + tol for v in per_x),
        "theorem": "Research.P8.IVInequality.instrumental_inequality",
    }


def misclassified_rd(r1: float, r0: float, prevalence: float, sensitivity: float, specificity: float) -> dict:
    """Nondifferential misclassification of a binary exposure.

    The measured risk difference is ``(se + sp - 1)(r1 - r0) pi (1 - pi) / (d1 d0)`` with
    ``d1, d0`` the measured group shares (Lean
    ``Research.P2.Misclassification.binary_attenuation``), so it keeps the sign of the true
    difference whenever ``se + sp >= 1``.

    >>> round(misclassified_rd(0.3, 0.1, 0.4, 0.8, 0.9)["factor"], 6)
    0.713073
    """
    if prevalence <= 0 or prevalence >= 1:
        raise ValueError("prevalence must be in (0, 1)")
    if not (0 <= sensitivity <= 1 and 0 <= specificity <= 1):
        raise ValueError("sensitivity and specificity must be in [0, 1]")
    pi, se, sp = prevalence, sensitivity, specificity
    d1 = pi * se + (1 - pi) * (1 - sp)
    d0 = pi * (1 - se) + (1 - pi) * sp
    if d1 <= 0 or d0 <= 0:
        raise ValueError("both measured groups must be non-empty")
    n1 = pi * se * r1 + (1 - pi) * (1 - sp) * r0
    n0 = pi * (1 - se) * r1 + (1 - pi) * sp * r0
    measured = n1 / d1 - n0 / d0
    true_rd = r1 - r0
    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731
    return {
        "true_rd": true_rd,
        "measured_rd": measured,
        "factor": (se + sp - 1) * pi * (1 - pi) / (d1 * d0),
        "d1": d1,
        "d0": d0,
        "sign_kept": sgn(measured) == sgn(true_rd) or measured == 0,
        "theorem": "Research.P2.Misclassification.binary_attenuation",
    }


def misclassified_contrast(risks, group, weights=None, high: str = "high", low: str = "low") -> dict:
    """Misclassification of an ordinal exposure can reverse a contrast.

    Levels are mapped to measured groups; the measured contrast of weighted mean risks can be
    zero or of the opposite sign to the true top-versus-bottom contrast when the map is not
    monotone (Lean ``Research.P2.Misclassification.ordinal_witness``, ``ordinal_reversal``).

    >>> misclassified_contrast([0, 0.5, 1], ["high", "low", "high"])["measured_contrast"]
    0.0
    """
    risks = [float(x) for x in risks]
    group = list(group)
    weights = [1.0] * len(risks) if weights is None else [float(x) for x in weights]
    if len(risks) < 2 or len(group) != len(risks) or len(weights) != len(risks):
        raise ValueError("risks, group and weights must have the same length, at least 2")
    if high not in group or low not in group:
        raise ValueError("high and low must both appear in group")
    gr: dict = {}
    for g in set(group):
        num = sum(w * r for w, r, gg in zip(weights, risks, group) if gg == g)
        den = sum(w for w, gg in zip(weights, group) if gg == g)
        gr[g] = num / den
    true_c = risks[-1] - risks[0]
    meas = gr[high] - gr[low]
    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731
    return {
        "true_contrast": true_c,
        "measured_contrast": meas,
        "group_risk": gr,
        "reversed": sgn(meas) != sgn(true_c),
        "theorem": "Research.P2.Misclassification.ordinal_witness",
    }


def sensitivity_amplification(Lambda=None, Delta=None, Gamma=None) -> dict:
    """Amplification of a sensitivity parameter: one Gamma is a curve of (Lambda, Delta).

    ``Gamma = (Lambda Delta + 1) / (Lambda + Delta)``; every ``Gamma >= 1`` is reached along
    ``Delta = (Lambda Gamma - 1) / (Lambda - Gamma)`` for ``Lambda > Gamma`` (Lean
    ``Research.P11.Amplification.level_curve``). Give any two of the three.

    >>> sensitivity_amplification(Gamma=2, Lambda=[3, 4])["Delta"]
    [5.0, 3.5]
    """
    given = sum(v is not None for v in (Lambda, Delta, Gamma))
    if given != 2:
        raise ValueError("give exactly two of Lambda, Delta, Gamma")

    def vec(v):
        return [float(x) for x in (v if isinstance(v, list | tuple) else [v])]

    def pair(a, b):
        a, b = vec(a), vec(b)
        n = max(len(a), len(b))
        return [a[i % len(a)] for i in range(n)], [b[i % len(b)] for i in range(n)]

    if Gamma is None:
        L, D = pair(Lambda, Delta)
        if any(x < 1 for x in L + D):
            raise ValueError("Lambda and Delta must be at least 1")
        G = [(lam * d + 1) / (lam + d) for lam, d in zip(L, D)]
    elif Delta is None:
        G, L = pair(Gamma, Lambda)
        if any(g < 1 for g in G) or any(lam <= g for lam, g in zip(L, G)):
            raise ValueError("Lambda must exceed Gamma (at least 1)")
        D = [(lam * g - 1) / (lam - g) for lam, g in zip(L, G)]
    else:
        G, D = pair(Gamma, Delta)
        if any(g < 1 for g in G) or any(d <= g for d, g in zip(D, G)):
            raise ValueError("Delta must exceed Gamma (at least 1)")
        L = [(d * g - 1) / (d - g) for d, g in zip(D, G)]
    return {"Lambda": L, "Delta": D, "Gamma": G, "theorem": "Research.P11.Amplification.level_curve"}


def design_effect_direction(rho: float, M: int, x=None) -> dict:
    """The design effect of clustering is direction-specific.

    The quadratic form of a regressor is ``(1 - rho) sum x^2 + rho sum_c (sum_j x_cj)^2`` (Lean
    ``Research.P13.Design.quad_form``): a cluster-level regressor inflates the naive variance by
    ``1 + (M - 1) rho``, a within-cluster-balanced one deflates it by ``1 - rho``, and
    ``rho >= -1/(M - 1)``.

    >>> design_effect_direction(0.1, 20)["deff_cluster_level"]
    2.9000000000000004
    """
    if M < 2:
        raise ValueError("M must be at least 2")
    floor_rho = -1 / (M - 1)
    if rho < floor_rho or rho > 1:
        raise ValueError("rho must lie in [-1/(M - 1), 1]")
    out = {
        "deff_cluster_level": 1 + (M - 1) * rho,
        "deff_within_balanced": 1 - rho,
        "icc_floor": floor_rho,
        "theorem": "Research.P13.Design.quad_form",
    }
    if x is not None:
        rows = [[float(v) for v in row] for row in x]
        if any(len(row) != M for row in rows):
            raise ValueError("x must have M columns")
        naive = sum(v * v for row in rows for v in row)
        out["quad"] = (1 - rho) * naive + rho * sum(sum(row) ** 2 for row in rows)
        out["naive"] = naive
        out["ratio"] = out["quad"] / naive
    return out


def ht_elephant(t, pi, bound: float | None = None) -> dict:
    """Basu's elephant: the Horvitz-Thompson total is unbiased with exploding variance.

    Mean ``sum t_i`` and variance ``sum t_i^2 (1 - pi_i) / pi_i`` under independent inclusion
    (Lean ``Research.P3.Basu.ht_variance``); the largest unit alone exceeds any bound ``B`` at
    ``pi_e = t_e^2 / (2 (B + t_e^2))`` (``elephant_unbounded``).

    >>> ht_elephant([1, 2, 3, 100], [0.5, 0.5, 0.5, 0.01])["mean"]
    106.0
    """
    t = [float(v) for v in t]
    pi = [float(v) for v in pi]
    if len(t) != len(pi) or any(v <= 0 or v > 1 for v in pi):
        raise ValueError("pi must be in (0, 1], one per unit")
    terms = [a * a * (1 - b) / b for a, b in zip(t, pi)]
    out = {"mean": sum(t), "variance": sum(terms), "terms": terms, "theorem": "Research.P3.Basu.ht_variance"}
    if bound is not None:
        if bound <= 0:
            raise ValueError("bound must be positive")
        e = max(range(len(t)), key=lambda i: abs(t[i]))
        te = t[e]
        pe = te * te / (2 * (bound + te * te))
        out["elephant"] = e
        out["pi_elephant"] = pe
        out["term_elephant"] = te * te * (1 - pe) / pe
    return out


def attractiveness_contraction(S, omega: float, eta: float, B, n: int = 10) -> dict:
    """The discrete attractiveness field contracts absent crime.

    ``B' = (1 - omega)[(1 - eta) B + eta S B]`` with ``S`` row-stochastic contracts the sup norm
    by ``1 - omega`` per step (Lean ``Research.P4.Contraction.decay``); with ``S`` doubly
    stochastic the total is scaled by exactly ``1 - omega``; the uniform feedback mode grows iff
    ``eps rho > omega / (1 - omega)``.

    >>> S = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    >>> attractiveness_contraction(S, 0.2, 0.3, [5, 1, 0], 2)["total"]
    [6.0, 4.800000000000001, 3.84]
    """
    S = [[float(v) for v in row] for row in S]
    V = len(S)
    if (
        any(len(row) != V for row in S)
        or any(v < 0 for row in S for v in row)
        or any(abs(sum(row) - 1) > 1e-10 for row in S)
    ):
        raise ValueError("S must be square, non-negative, with unit row sums")
    B = [float(v) for v in B]
    if len(B) != V:
        raise ValueError("B must have one value per site")
    if not (0 <= omega <= 1 and 0 <= eta <= 1):
        raise ValueError("omega and eta must lie in [0, 1]")
    traj = [list(B)]
    for _ in range(int(n)):
        SB = [sum(S[i][j] * B[j] for j in range(V)) for i in range(V)]
        B = [(1 - omega) * ((1 - eta) * B[i] + eta * SB[i]) for i in range(V)]
        traj.append(list(B))
    sup = [max(abs(v) for v in row) for row in traj]
    return {
        "trajectory": traj,
        "sup_norm": sup,
        "sup_bound": [(1 - omega) ** k * sup[0] for k in range(len(traj))],
        "total": [sum(row) for row in traj],
        "doubly_stochastic": all(abs(sum(S[i][j] for i in range(V)) - 1) < 1e-10 for j in range(V)),
        "feedback_threshold": omega / (1 - omega) if omega < 1 else math.inf,
        "theorem": "Research.P4.Contraction.decay",
    }


def inspection_game(a: float, x, steps: int = 1000, dt: float = 0.01) -> dict:
    """The criminal-citizen-inspector game cycles.

    Replicator dynamics for a rock-paper-scissors payoff with stake ``a`` has ``x^T A x = 0``,
    the centroid as its only interior rest point and ``V = x1 x2 x3`` conserved (Lean
    ``Research.P8.Inspection.conserved``): orbits are closed curves around the centroid.

    >>> inspection_game(1, [1/3, 1/3, 1/3], 5)["field"]
    [0.0, 0.0, 0.0]
    """
    if a == 0:
        raise ValueError("a must be non-zero")
    x = [float(v) for v in x]
    if len(x) != 3 or any(v < 0 for v in x) or abs(sum(x) - 1) > 1e-10:
        raise ValueError("x must be three shares summing to one")

    def field(x):
        return [x[0] * a * (x[1] - x[2]), x[1] * a * (x[2] - x[0]), x[2] * a * (x[0] - x[1])]

    f0 = field(x)
    dv = x[1] * x[2] * f0[0] + x[0] * x[2] * f0[1] + x[0] * x[1] * f0[2]
    traj = [list(x)]
    for _ in range(int(steps)):
        k1 = field(x)
        k2 = field([x[i] + dt / 2 * k1[i] for i in range(3)])
        k3 = field([x[i] + dt / 2 * k2[i] for i in range(3)])
        k4 = field([x[i] + dt * k3[i] for i in range(3)])
        x = [x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
        traj.append(list(x))
    v = [row[0] * row[1] * row[2] for row in traj]
    return {
        "trajectory": traj,
        "invariant": v,
        "invariant_drift": max(abs(u - v[0]) for u in v),
        "dV_dt": dv,
        "field": f0,
        "fixed_point": [1 / 3, 1 / 3, 1 / 3],
        "theorem": "Research.P8.Inspection.conserved",
    }
