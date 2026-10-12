# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research ledger, the third library-scan batch of 2026-10-12: the Python side of twenty results.

Every statement is a machine-checked theorem (Lean 4 + Mathlib, 0 sorry, standard axioms only)
in ``research/lean/Researchproofs/`` of the R package (rmorie); the names mirror the R functions
``morie_*`` without the prefix. Each result carries the theorem it computes from as ``theorem``.
"""

from __future__ import annotations

import itertools
import math

__all__ = [
    "nb_matches_zip",
    "gamma_bound",
    "blocking_crossover",
    "adjustment_variances",
    "budget_optimum",
    "outcome_test",
    "spatial_omission",
    "iv_natural_bounds",
    "design_sensitivity",
    "moran_trend",
    "exposure_positivity",
    "permutation_valid",
    "thinning_identification",
    "poisson_chernoff",
    "random_walk_exposure",
    "extremal_index_clusters",
    "tail_ratio",
    "calibration_flip",
    "bayes_risk_dpi",
    "bellman_value",
    "myopic_example",
    "budget_rule",
    "run_length_bound",
]


def _bisect(f, lo: float, hi: float, tol: float = 1e-12, it: int = 400) -> float:
    flo = f(lo)
    for _ in range(it):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _t_quantile(p: float, df: float) -> float:
    """Upper-tail t quantile: bisection on the CDF, the density integrated by Simpson's rule."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def dens(t):
        return c * (1 + t * t / df) ** (-(df + 1) / 2)

    def cdf(t):
        if t < 0:
            return 1 - cdf(-t)
        n = 4000
        h = t / n
        acc = dens(0) + dens(t)
        for i in range(1, n):
            acc += (4 if i % 2 else 2) * dens(i * h)
        return 0.5 + acc * h / 3

    return _bisect(lambda t: cdf(t) - p, 0, 50, tol=1e-10)


def nb_matches_zip(pi: float, lam: float) -> dict:
    """A negative binomial with the zero share and mean of any zero-inflated Poisson.

    Lean ``Research.P7.ZeroInflation.nb_matches_zip``.

    >>> r = nb_matches_zip(0.3, 2)
    >>> abs(r["nb_zero"] - r["zip_zero"]) < 1e-9
    True
    """
    if pi <= 0 or pi >= 1:
        raise ValueError("pi must be in (0, 1)")
    if lam <= 0:
        raise ValueError("lam must be positive")
    mu = (1 - pi) * lam
    p0 = pi + (1 - pi) * math.exp(-lam)
    theta = _bisect(lambda th: (1 + mu / th) ** (-th) - p0, 1e-8, 1e8)
    return {
        "mean": mu,
        "zip_zero": p0,
        "theta": theta,
        "nb_zero": (1 + mu / theta) ** (-theta),
        "size": theta,
        "theorem": "Research.P7.ZeroInflation.nb_matches_zip",
    }


def gamma_bound(T: float, n: float, Gamma=None) -> dict:
    """A Rosenbaum Gamma bound is not a test of ignorability (Lean ``Research.P5.Gamma.gamma_star_kills``).

    >>> gamma_bound(70, 100)["gamma_star"]
    2.3333333333333335
    """
    if T < 0 or n <= T:
        raise ValueError("T must satisfy 0 <= T < n")
    out = {"gamma_star": T / (n - T), "theorem": "Research.P5.Gamma.gamma_star_kills"}
    if Gamma is not None:
        G = [float(g) for g in Gamma]
        if any(g < 0 for g in G):
            raise ValueError("Gamma must be non-negative")
        wm = [n * g / (1 + g) for g in G]
        out["worst"] = {"Gamma": G, "worst_mean": wm, "survives": [w < T for w in wm]}
    return out


def blocking_crossover(delta: float, sigma: float, n: int, alpha: float = 0.05, rho=None) -> dict:
    """The within-pair correlation crossover for blocking (Lean ``Research.P3.Blocking.paired_better_iff``).

    >>> blocking_crossover(0.5, 1, 10)["crossover"] > 0
    True
    """
    if delta <= 0 or sigma <= 0 or n < 3:
        raise ValueError("delta, sigma positive and n at least 3")
    k = delta * math.sqrt(n / 2) / sigma
    cp = _t_quantile(1 - alpha, n - 1)
    cu = _t_quantile(1 - alpha, 2 * n - 2)
    out = {
        "k": k,
        "c_paired": cp,
        "c_unpaired": cu,
        "crossover": 1 - 1 / (1 + (cp - cu) / k) ** 2,
        "theorem": "Research.P3.Blocking.paired_better_iff",
    }
    if rho is not None:
        R = [float(x) for x in rho]
        if any(x >= 1 for x in R):
            raise ValueError("rho must be below 1")
        out["z_paired"] = [k / math.sqrt(1 - x) - cp for x in R]
        out["z_unpaired"] = k - cu
        out["paired_better"] = [z > k - cu for z in out["z_paired"]]
    return out


def adjustment_variances(b1: float, b0: float, v1: float, v0: float, p: float) -> dict:
    """Freedman/Lin asymptotic variances (Lean ``Research.P3.Adjustment.adjusted_minus_unadjusted``).

    >>> adjustment_variances(1, 0, 1, 1, 0.8)["adjustment_hurts"]
    True
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    if v1 < 0 or v0 < 0:
        raise ValueError("noise variances must be non-negative")
    d = b1 - b0
    un = (b1 * b1 + v1) / p + (b0 * b0 + v0) / (1 - p)
    ad = ((1 - p) ** 2 * d * d + v1) / p + (p * p * d * d + v0) / (1 - p)
    lin = v1 / p + v0 / (1 - p)
    return {
        "unadjusted": un,
        "adjusted": ad,
        "lin": lin,
        "adjusted_minus_unadjusted": ad - un,
        "adjustment_hurts": ad > un,
        "theorem": "Research.P3.Adjustment.adjusted_minus_unadjusted",
    }


def budget_optimum(g, d, C: float, grid: int = 201) -> dict:
    """Stick versus carrot: the interior optimum on a grid (Lean ``Research.P8.Budget.interior_optimum``).

    >>> budget_optimum(math.sqrt, lambda s: 2 * math.log1p(s), 10)["interior"]
    True
    """
    if C <= 0:
        raise ValueError("C must be positive")
    theta = [C * i / (grid - 1) for i in range(grid)]
    f = [g(t) + d(C - t) for t in theta]
    i = max(range(grid), key=lambda j: f[j])
    return {
        "theta": theta[i],
        "value": f[i],
        "interior": 0 < i < grid - 1,
        "corner_values": {"stick": f[0], "carrot": f[-1]},
        "grid": {"theta": theta, "objective": f},
        "theorem": "Research.P8.Budget.interior_optimum",
    }


def outcome_test(groups: dict) -> dict:
    """Stop and hit rates of threshold groups (Lean ``Research.P2.OutcomeTest.infra_marginality``).

    >>> r = outcome_test({"A": {"q": [0.3, 0.9], "w": [1, 1], "t": 0.5}, "B": {"q": [0.9], "w": [1], "t": 0.2}})
    >>> r["A"]["hit_rate"] == r["B"]["hit_rate"]
    True
    """
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a non-empty dict")
    out = {}
    for name, g in groups.items():
        q = [float(x) for x in g["q"]]
        w = [float(x) for x in g["w"]]
        t = float(g["t"])
        if any(x < 0 or x > 1 for x in q) or any(x < 0 for x in w) or sum(w) <= 0:
            raise ValueError("each group needs q in [0, 1] and positive weights")
        s = [qi > t for qi in q]
        stopped = sum(wi for wi, si in zip(w, s) if si)
        stop_rate = stopped / sum(w)
        hit = sum(wi * qi for wi, qi, si in zip(w, q, s) if si) / stopped if stopped > 0 else float("nan")
        out[name] = {"threshold": t, "stop_rate": stop_rate, "hit_rate": hit}
    out["theorem"] = "Research.P2.OutcomeTest.infra_marginality"
    return out


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-300:
            raise ValueError("singular matrix")
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def spatial_omission(rho: float, x, Sigma, alpha: float = 0.05, m: int = 1) -> dict:
    """Lag endogeneity, GLS over OLS, and the family-wise error (Lean ``Research.P3.Spatial.lag_regressor_correlated``).

    >>> r = spatial_omission(0.4, [1, 2], [[1, 0.5], [0.5, 1]])
    >>> (round(r["var_ols"], 6), round(r["var_gls"], 6))
    (0.28, 0.25)
    """
    if rho * rho == 1:
        raise ValueError("rho^2 must differ from 1")
    x = [float(v) for v in x]
    S = [[float(v) for v in row] for row in Sigma]
    if len(x) != len(S) or any(len(row) != len(S) for row in S):
        raise ValueError("Sigma must be square with len(x) rows")
    xSx = sum(x[i] * S[i][j] * x[j] for i in range(len(x)) for j in range(len(x)))
    vo = xSx / sum(v * v for v in x) ** 2
    Sinv_x = _solve(S, x)
    vg = 1 / sum(xi * yi for xi, yi in zip(x, Sinv_x))
    return {
        "lag_covariance": rho / (1 - rho * rho),
        "var_ols": vo,
        "var_gls": vg,
        "gls_more_precise": vg <= vo + 1e-12,
        "fwer": 1 - (1 - alpha) ** m,
        "theorem": "Research.P3.Spatial.lag_regressor_correlated",
    }


def iv_natural_bounds(p11: float, p10: float, m1: float, m0: float) -> dict:
    """IV natural bounds (Lean ``Research.P8.NaturalBounds.width``).

    >>> round(iv_natural_bounds(0.8, 0.3, 0.6, 0.4)["width"], 12)
    0.5
    """
    if any(v < 0 or v > 1 for v in (p11, p10, m1, m0)):
        raise ValueError("all inputs must lie in [0, 1]")
    lower = m1 * p11 - m0 * (1 - p10) - p10
    upper = m1 * p11 + (1 - p11) - m0 * (1 - p10)
    return {"lower": lower, "upper": upper, "width": upper - lower, "theorem": "Research.P8.NaturalBounds.width"}


def design_sensitivity(mu: float, mu_gamma=None, upper: float = 1e4) -> dict:
    """Design sensitivity as the root of mu_Gamma = mu (Lean ``Research.P5.DesignSensitivity.design_sensitivity_exists_unique``).

    >>> round(design_sensitivity(0.7)["gamma_tilde"], 6)
    2.333333
    """
    if mu_gamma is None:
        mu_gamma = lambda G: G / (1 + G)  # noqa: E731
    if mu_gamma(1) >= mu:
        raise ValueError("mu must exceed mu_gamma(1): no effect to detect")
    if mu_gamma(upper) <= mu:
        raise ValueError("upper too small: mu_gamma(upper) must exceed mu")
    gt = _bisect(lambda G: mu_gamma(G) - mu, 1, upper, tol=1e-10)
    G = [1, gt / 2 + 0.5, 2 * gt]
    return {
        "gamma_tilde": gt,
        "limiting_power": {"Gamma": G, "power": [1.0 if mu_gamma(g) < mu else 0.0 for g in G]},
        "theorem": "Research.P5.DesignSensitivity.design_sensitivity_exists_unique",
    }


def moran_trend(W, y, X=None) -> dict:
    """Moran's I under a trend and on residuals (Lean ``Research.P7.MoranTrend.residual_moran_ratio``).

    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(moran_trend(W, [1, 2, 3, 4])["I"], 6)
    0.333333
    """
    W = [[float(v) for v in row] for row in W]
    y = [float(v) for v in y]
    n = len(y)
    if len(W) != n or any(len(row) != n for row in W):
        raise ValueError("W must be n x n")
    S0 = sum(sum(row) for row in W)
    ybar = sum(y) / n
    z = [v - ybar for v in y]
    quad = sum(z[i] * W[i][j] * z[j] for i in range(n) for j in range(n))
    out = {
        "I": (n / S0) * quad / sum(v * v for v in z),
        "null_expectation": -1 / (n - 1),
        "theorem": "Research.P7.MoranTrend.residual_moran_ratio",
    }
    if X is not None:
        X = [[float(v) for v in row] for row in X]
        p = len(X[0])
        XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
        # H = X (X'X)^{-1} X'
        cols = [[X[k][j] for k in range(n)] for j in range(p)]
        H = [[0.0] * n for _ in range(n)]
        for i in range(n):
            coef = _solve(XtX, [X[i][j] for j in range(p)])
            for k in range(n):
                H[i][k] = sum(coef[j] * cols[j][k] for j in range(p))
        M = [[(1.0 if i == k else 0.0) - H[i][k] for k in range(n)] for i in range(n)]
        e = [sum(M[i][k] * y[k] for k in range(n)) for i in range(n)]
        out["I_residual"] = (
            (n / S0) * sum(e[i] * W[i][j] * e[j] for i in range(n) for j in range(n)) / sum(v * v for v in e)
        )
        trWM = sum(W[i][k] * M[k][i] for i in range(n) for k in range(n))
        out["residual_expectation"] = (n / S0) * trWM / (n - p)
    return out


def exposure_positivity(p: float, degrees) -> dict:
    """Exposure positivity under Bernoulli assignment (Lean ``Research.P3.Exposure.exposure_prob_tendsto_zero``).

    >>> exposure_positivity(0.5, [1, 2])["ht_weight"]
    [2.0, 4.0]
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    D = [int(d) for d in degrees]
    if any(d < 0 for d in D):
        raise ValueError("degrees must be non-negative")
    return {
        "degree": D,
        "prob_all_treated": [p**d for d in D],
        "ht_weight": [p ** (-d) for d in D],
        "theorem": "Research.P3.Exposure.exposure_prob_tendsto_zero",
    }


def permutation_valid(stats, alpha=(0.01, 0.05, 0.1)) -> dict:
    """Exactness of the permutation p-value (Lean ``Research.P3.Exposure.permutation_valid``).

    >>> permutation_valid([3, 1, 4, 1, 5], [0.5])["levels"]["valid"]
    [True]
    """
    s = [float(v) for v in stats]
    if not s:
        raise ValueError("stats must be non-empty")
    A = [float(a) for a in alpha]
    if any(a < 0 or a > 1 for a in A):
        raise ValueError("alpha must lie in [0, 1]")
    pv = [sum(1 for u in s if u >= t) / len(s) for t in s]
    share = [sum(1 for v in pv if v <= a) / len(s) for a in A]
    return {
        "pvalues": pv,
        "levels": {"alpha": A, "rejection_share": share, "valid": [sh <= a + 1e-12 for sh, a in zip(share, A)]},
        "theorem": "Research.P3.Exposure.permutation_valid",
    }


def thinning_identification(lam: float, gamma: float, s=(0.0, 0.5, 0.9, 1.0)) -> dict:
    """Thinning a Poisson count (Lean ``Research.P1.Thinning.not_identified``).

    >>> thinning_identification(5, 0.3)["recorded_rate"]
    1.5
    """
    if lam <= 0:
        raise ValueError("lam must be positive")
    if gamma <= 0 or gamma >= 1:
        raise ValueError("gamma must be in (0, 1)")
    ok = all(abs(math.exp(lam * ((1 - gamma + gamma * v) - 1)) - math.exp(lam * gamma * (v - 1))) < 1e-12 for v in s)
    return {
        "recorded_rate": lam * gamma,
        "alternative": {"lam": lam * gamma, "gamma": 1.0},
        "pgf_check": ok,
        "theorem": "Research.P1.Thinning.not_identified",
    }


def _ppois_upper(k: int, lam: float) -> float:
    """P(Poisson(lam) >= k)."""
    term = math.exp(-lam)
    cdf = 0.0
    for j in range(k):
        cdf += term
        term *= lam / (j + 1)
    return max(0.0, 1 - cdf)


def poisson_chernoff(lam: float, eps) -> dict:
    """Chernoff tail bound for a Poisson count (Lean ``Research.P7.Chernoff.poisson_chernoff``).

    >>> r = poisson_chernoff(20, [0.5])
    >>> r["exact"][0] <= r["bound"][0]
    True
    """
    E = [float(e) for e in eps]
    if lam <= 0 or any(e <= 0 for e in E):
        raise ValueError("lam and eps must be positive")
    rate = [(1 + e) * math.log1p(e) - e for e in E]
    return {
        "eps": E,
        "bound": [math.exp(-lam * r) for r in rate],
        "exact": [_ppois_upper(math.ceil((1 + e) * lam), lam) for e in E],
        "rate": rate,
        "theorem": "Research.P7.Chernoff.poisson_chernoff",
    }


def random_walk_exposure(adj) -> dict:
    """Random-walk exposure is proportional to degree (Lean ``Research.P3.RandomWalk.stationary``).

    >>> random_walk_exposure([[0, 1, 1], [1, 0, 1], [1, 1, 0]])["stationary"]
    True
    """
    A = [[float(v) for v in row] for row in adj]
    n = len(A)
    if any(len(row) != n for row in A) or any(v < 0 for row in A for v in row):
        raise ValueError("adj must be a square non-negative matrix")
    d = [sum(row) for row in A]
    if any(v == 0 for v in d):
        raise ValueError("every vertex needs an out-edge")
    pi = [v / sum(d) for v in d]
    piP = [sum(pi[u] * A[u][v] / d[u] for u in range(n)) for v in range(n)]
    return {
        "pi": pi,
        "stationary": all(abs(a - b) < 1e-10 for a, b in zip(piP, pi)),
        "undirected": all(A[i][j] == A[j][i] for i in range(n) for j in range(n)),
        "theorem": "Research.P3.RandomWalk.stationary",
    }


def extremal_index_clusters(m: int, tau: float, n=(10, 100, 1000)) -> dict:
    """Extremal index of a block-cluster process (Lean ``Research.P10.ExtremalIndex.extremal_index``).

    >>> extremal_index_clusters(4, 2)["theta"]
    0.25
    """
    N = [int(v) for v in n]
    if m < 1 or tau <= 0 or any(v < 1 for v in N):
        raise ValueError("m, n at least 1 and tau positive")
    return {
        "theta": 1 / m,
        "limit_cluster": math.exp(-tau / m),
        "limit_iid": math.exp(-tau),
        "table": {
            "n": N,
            "cluster": [(1 - tau / (v * m)) ** v for v in N],
            "iid": [(1 - tau / (v * m)) ** (v * m) for v in N],
        },
        "theorem": "Research.P10.ExtremalIndex.extremal_index",
    }


def tail_ratio(k, q: float = 0.5, alpha: float = 1.0, r: int = 2, counts=None) -> dict:
    """Tail ratios separating light from power-law tails (Lean ``Research.P7.TailShape.nb_pmf_ratio_tendsto_zero``).

    >>> tail_ratio([5], q=0.7, alpha=1.5, r=3)["nb_pmf_ratio"][0] <= tail_ratio([5], q=0.7, alpha=1.5, r=3)["nb_bound"][0]
    True
    """
    K = [int(v) for v in k]
    if any(v < 1 for v in K) or q <= 0 or q >= 1 or alpha <= 0 or r < 1:
        raise ValueError("k, r at least 1, q in (0, 1), alpha positive")

    def nb(j):
        return math.comb(j + r - 1, j) * (1 - q) ** r * q**j

    out = {
        "k": K,
        "geometric": [q**v for v in K],
        "power": [2 ** (-alpha)] * len(K),
        "nb_pmf_ratio": [nb(2 * v) / nb(v) for v in K],
        "nb_bound": [(2 * v + r) ** (r - 1) * q**v for v in K],
        "theorem": "Research.P7.TailShape.nb_pmf_ratio_tendsto_zero",
    }
    if counts is not None:
        c = [float(v) for v in counts]

        def S(j):
            return sum(1 for v in c if v > j) / len(c)

        out["empirical"] = [S(2 * v) / S(v) if S(v) > 0 else float("nan") for v in K]
    return out


def calibration_flip(fA: float, fB: float, r) -> dict:
    """Calibration to the record and to the truth cannot both hold (Lean ``Research.P5.DataProcessing.calibration_incompatible``).

    >>> calibration_flip(0.05, 0.2, [0.3, 0.5])["incompatible"]
    [True, False]
    """
    if fA == 0.5 or fB == 0.5:
        raise ValueError("flip rates of 1/2 carry no information")
    R = [float(v) for v in r]
    return {
        "r": R,
        "true_A": [(v - fA) / (1 - 2 * fA) for v in R],
        "true_B": [(v - fB) / (1 - 2 * fB) for v in R],
        "recorded_A": [fA + (1 - 2 * fA) * v for v in R],
        "recorded_B": [fB + (1 - 2 * fB) * v for v in R],
        "incompatible": [fA != fB and v != 0.5 for v in R],
        "theorem": "Research.P5.DataProcessing.calibration_incompatible",
    }


def bayes_risk_dpi(mu, w, h) -> dict:
    """Post-processing cannot lower the best squared-error risk (Lean ``Research.P5.DataProcessing.bayes_risk_data_processing``).

    >>> bayes_risk_dpi([[0.3, 0.1, 0.05], [0.1, 0.2, 0.25]], [0, 1], [1, 1, 2])["dpi_holds"]
    True
    """
    M = [[float(v) for v in row] for row in mu]
    w = [float(v) for v in w]
    h = list(h)
    if len(w) != len(M) or any(len(row) != len(h) for row in M) or any(v < 0 for row in M for v in row):
        raise ValueError("w indexes rows, h columns, weights non-negative")

    def best(groups):
        risk = 0.0
        for g in set(groups):
            cols = [j for j, gg in enumerate(groups) if gg == g]
            mass = [sum(M[i][j] for j in cols) for i in range(len(w))]
            tot = sum(mass)
            if tot > 0:
                pred = sum(a * b for a, b in zip(mass, w)) / tot
                risk += sum(a * (b - pred) ** 2 for a, b in zip(mass, w))
        return risk

    rd = best(list(range(len(h))))
    rr = best(h)
    return {
        "risk_D": rd,
        "risk_R": rr,
        "dpi_holds": rd <= rr + 1e-12,
        "theorem": "Research.P5.DataProcessing.bayes_risk_data_processing",
    }


def bellman_value(r, p, gamma: float, iters: int = 1000, tol: float = 1e-10) -> dict:
    """Value iteration with the contraction audited (Lean ``Research.P4.Bellman.bellman_contracting``).

    >>> r = [[1, 0], [0.5, 0.5], [1, 1]]
    >>> p = [[[0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]]]
    >>> bellman_value(r, p, 0.9)["policy"][0]
    1
    """
    R = [[float(v) for v in row] for row in r]
    P = [[[float(v) for v in pr] for pr in row] for row in p]
    if gamma < 0 or gamma >= 1:
        raise ValueError("gamma must lie in [0, 1)")
    nS, nA = len(R), len(R[0])
    if len(P) != nS or any(len(row) != nA for row in P) or any(len(pr) != nS for row in P for pr in row):
        raise ValueError("p must be states x actions x states")
    if any(v < 0 for row in P for pr in row for v in pr) or any(abs(sum(pr) - 1) > 1e-10 for row in P for pr in row):
        raise ValueError("p rows must be probability vectors")
    v = [0.0] * nS
    ratio, prev, step, policy, i = 0.0, None, float("inf"), [0] * nS, 0
    for it in range(1, iters + 1):
        i = it
        q = [[R[s][a] + gamma * sum(P[s][a][t] * v[t] for t in range(nS)) for a in range(nA)] for s in range(nS)]
        nv = [max(row) for row in q]
        policy = [max(range(nA), key=lambda a, row=row: row[a]) for row in q]
        step = max(abs(a - b) for a, b in zip(nv, v))
        if prev is not None and prev > 1e-6:
            ratio = max(ratio, step / prev)
        v = nv
        if step < tol:
            break
        prev = step
    return {
        "value": v,
        "policy": policy,
        "iterations": i,
        "contraction_ratio": ratio,
        "converged": step < tol,
        "theorem": "Research.P4.Bellman.bellman_contracting",
    }


def myopic_example(gamma) -> dict:
    """Greedy versus patient allocation (Lean ``Research.P4.Bellman.myopic_suboptimal``).

    >>> myopic_example([0.5, 0.9])["patient_better"]
    [False, True]
    """
    G = [float(g) for g in gamma]
    if any(g < 0 or g >= 1 for g in G):
        raise ValueError("gamma must lie in [0, 1)")
    gr = [1 + g * 0.5 / (1 - g) for g in G]
    pt = [g / (1 - g) for g in G]
    return {
        "gamma": G,
        "greedy": gr,
        "patient": pt,
        "patient_better": [b > a for a, b in zip(gr, pt)],
        "theorem": "Research.P4.Bellman.myopic_suboptimal",
    }


def budget_rule(m, tau, kappa: float) -> dict:
    """The budget-constrained threshold rule (Lean ``Research.P5.BudgetRule.threshold_optimal``).

    >>> budget_rule([1, 1, 1], [0.5, 0.2, -0.1], 1.5)["rule"]
    [1.0, 0.5, 0.0]
    """
    M = [float(v) for v in m]
    T = [float(v) for v in tau]
    if len(M) != len(T) or any(v <= 0 for v in M) or kappa < 0:
        raise ValueError("m positive, same length as tau, kappa non-negative")
    order = sorted(range(len(M)), key=lambda i: -T[i])
    d = [0.0] * len(M)
    left, thr = kappa, 0.0
    for i in order:
        if T[i] <= 0:
            break
        take = min(1.0, left / M[i])
        d[i] = take
        left -= take * M[i]
        thr = T[i]
        if take < 1:
            break
    value = sum(a * b * c for a, b, c in zip(M, d, T))
    out = {"rule": d, "threshold": thr, "value": value, "theorem": "Research.P5.BudgetRule.threshold_optimal"}
    if len(M) <= 15:
        best = max(
            (
                sum(gi * mi * ti for gi, mi, ti in zip(gvec, M, T))
                for gvec in itertools.product((0, 1), repeat=len(M))
                if sum(gi * mi for gi, mi in zip(gvec, M)) <= kappa + 1e-12
            ),
            default=0.0,
        )
        out["best_deterministic"] = best
        out["stochastic_needed"] = best < value - 1e-12
    return out


def run_length_bound(alpha: float, J: int | None = None) -> dict:
    """Mean run length of a restarted detector (Lean ``Research.P10.Cusum.mean_run_length_ge``).

    >>> run_length_bound(0.01)["bound"]
    50.0
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    if J is None:
        J = int(math.floor(1 / alpha))
    if J * alpha > 1:
        raise ValueError("J * alpha must not exceed 1")
    return {
        "J": J,
        "partial_sum": sum(1 - n * alpha for n in range(J)),
        "bound": J / 2,
        "theorem": "Research.P10.Cusum.mean_run_length_ge",
    }
