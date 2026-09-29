# morie.fn -- function file (rootcoder007/morie)
"""Spatial epidemiology: Bayesian zero-inflated Poisson rates by Gibbs sampling, Kelsall-Diggle
kernel relative risk, Oden's population-adjusted Moran statistic, Kulldorff's prospective
space-time scan, Poisson ecological regression, areal wombling (barriers), point-source buffer
rate ratios, need-based allocation, funnel-plot control limits and the Ghose drug-likeness filter."""

from __future__ import annotations

import math

from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform

__all__ = [
    "zip_gibbs",
    "kernel_relative_risk",
    "oden_ipop",
    "prospective_scan",
    "poisson_ecological",
    "areal_wombling",
    "buffer_rate_ratio",
    "need_based_allocation",
    "funnel_control_limits",
    "ghose_drug_filter",
]


def _rgamma(shape, seed, stream):
    # Marsaglia-Tsang (2000) on Philox blocks; shape < 1 by the boost G(a) = G(a + 1) U^(1/a)
    boost = 1.0
    a = shape
    if a < 1.0:
        u0 = float(random_uniform(1, seed=seed, stream=2 * (stream * 64 + 63))[0])
        boost = u0 ** (1.0 / a)
        a += 1.0
    d = a - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    for blk in range(63):
        z = random_normal(8, seed=seed, stream=2 * (stream * 64 + blk))
        u = random_uniform(8, seed=seed, stream=2 * (stream * 64 + blk) + 1)
        for q in range(8):
            x = float(z[q])
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            if math.log(float(u[q])) < 0.5 * x * x + d - d * v + d * math.log(v):
                return d * v * boost
    return d * boost


def zip_gibbs(y, expected, n_iter: int, *, a: float = 1.0, b: float = 1.0, burn: int = 0, seed: int = 0) -> RichResult:
    r"""Bayesian zero-inflated Poisson relative risks by Gibbs sampling with data augmentation.

    ``y_i ~ pi 1(0) + (1 - pi) Poisson(E_i theta_i)``, ``theta_i ~ Gamma(a, b)``,
    ``pi ~ Beta(1, 1)``. Each sweep draws the structural-zero indicators
    ``z_i | y_i = 0 ~ Bernoulli(pi / (pi + (1 - pi) exp(-E_i theta_i)))``,
    ``theta_i ~ Gamma(a + y_i (1 - z_i), b + E_i (1 - z_i))`` and
    ``pi ~ Beta(1 + sum z, 1 + n - sum z)`` (two gammas). Gammas use
    Marsaglia-Tsang (gamma draw ``k`` of sweep ``t`` on Philox streams
    ``2 (64 (t (n + 3) + k) + b)`` and ``+ 1`` for blocks ``b``; indicator
    uniforms on seed ``seed + 7919``, stream ``t``); returns posterior means
    after ``burn``.

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression. Technometrics 34,
    1-14. Ghosh, S. K., Mukhopadhyay, P. and Lu, J.-C. (2006). Bayesian
    analysis of zero-inflated regression models. JSPI 136, 1360-1375.
    Marsaglia, G. and Tsang, W. W. (2000). ACM TOMS 26, 363-372.

    Examples
    --------
    >>> r = zip_gibbs([0, 0, 3, 5, 0, 2], [1.0, 1.2, 2.0, 3.1, 0.8, 1.5], 50, seed=1)
    >>> 0.0 < r.pi < 1.0, len(r.theta)
    (True, 6)
    """
    ys = [int(v) for v in y]
    E = [float(v) for v in expected]
    n = len(ys)
    theta = [1.0] * n
    pi = 0.5
    z = [0] * n
    keep = 0
    s_theta = [0.0] * n
    s_pi = 0.0
    for t in range(n_iter):
        base = t * (n + 3)
        u = random_uniform(n, seed=seed + 7919, stream=t)
        for i in range(n):
            if ys[i] == 0:
                p0 = pi / (pi + (1 - pi) * math.exp(-E[i] * theta[i]))
                z[i] = 1 if float(u[i]) < p0 else 0
            else:
                z[i] = 0
        for i in range(n):
            g = _rgamma(a + ys[i] * (1 - z[i]), seed, base + i)
            theta[i] = g / (b + E[i] * (1 - z[i]))
        sz = sum(z)
        g1 = _rgamma(1.0 + sz, seed, base + n)
        g2 = _rgamma(1.0 + n - sz, seed, base + n + 1)
        pi = g1 / (g1 + g2)
        if t >= burn:
            keep += 1
            s_pi += pi
            for i in range(n):
                s_theta[i] += theta[i]
    return RichResult(payload={"theta": [v / keep for v in s_theta], "pi": s_pi / keep, "draws": keep})


def kernel_relative_risk(cases, controls, points, bandwidth: float) -> RichResult:
    r"""Kelsall-Diggle kernel estimate of the log relative risk surface.

    ``rho(x) = log(f_1(x) / f_0(x))`` with Gaussian kernel density estimates
    (common bandwidth ``h``) of the case and control point patterns, each
    normalised by its size, evaluated at ``points``; also returns the case
    probability ``lambda_1 / (lambda_1 + lambda_0)`` of the intensities.

    References
    ----------
    Kelsall, J. E. and Diggle, P. J. (1995). Non-parametric estimation of
    spatial variation in relative risk. Statistics in Medicine 14, 2335-2342.
    Bithell, J. F. (1990). An application of density estimation to
    geographical epidemiology. Statistics in Medicine 9, 691-701.

    Examples
    --------
    >>> r = kernel_relative_risk([(0.0, 0.0), (0.1, 0.0)], [(1.0, 1.0), (0.9, 1.0)], [(0.5, 0.5)], 0.5)
    >>> round(r.log_rr[0], 10)
    0.0
    """

    def dens(P, x):
        return ssum(math.exp(-((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2) / (2 * bandwidth**2)) for p in P) / (
            len(P) * 2 * math.pi * bandwidth**2
        )

    f1 = [dens(cases, x) for x in points]
    f0 = [dens(controls, x) for x in points]
    n1, n0 = len(cases), len(controls)
    return RichResult(
        payload={
            "log_rr": [math.log(a / c) for a, c in zip(f1, f0)],
            "f_cases": f1,
            "f_controls": f0,
            "p_case": [n1 * a / (n1 * a + n0 * c) for a, c in zip(f1, f0)],
        }
    )


def oden_ipop(cases, population, W, *, nsim: int = 0, seed: int = 0) -> RichResult:
    r"""Oden's population-adjusted Moran statistic ``I_pop`` with a conditional Monte Carlo test.

    With ``m`` the total population, ``b`` the overall rate, ``r_i`` the
    share of cases and ``p_i`` the share of population in region ``i``:
    ``I_pop = [m^2 sum w_ij (r_i - p_i)(r_j - p_j) - m (1 - 2b) sum w_ii r_i - m b sum w_ii p_i]
    / [b (1 - b) (m^2 sum w_ij p_i p_j - m sum w_ii p_i)]``.
    The Monte Carlo p-value redistributes the case total multinomially in
    proportion to population (Philox streams ``1..nsim``).

    References
    ----------
    Oden, N. (1995). Adjusting Moran's I for population density.
    Statistics in Medicine 14, 17-26. Waller, L. A. and Gotway, C. A.
    (2004). Applied Spatial Statistics for Public Health Data, section 7.4.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> r = oden_ipop([5, 5, 5], [100, 100, 100], W)
    >>> round(r.statistic, 12) + 0.0
    0.0
    """
    c = [float(v) for v in cases]
    pop = [float(v) for v in population]
    n = len(c)
    C, m = ssum(c), ssum(pop)
    b = C / m
    p = [v / m for v in pop]
    Wf = [[float(v) for v in r] for r in W]
    den = (
        b
        * (1 - b)
        * (
            m * m * ssum(Wf[i][j] * p[i] * p[j] for i in range(n) for j in range(n))
            - m * ssum(Wf[i][i] * p[i] for i in range(n))
        )
    )

    def stat(cc):
        tot = ssum(cc)
        r = [v / tot for v in cc]
        num = m * m * ssum(Wf[i][j] * (r[i] - p[i]) * (r[j] - p[j]) for i in range(n) for j in range(n))
        num -= m * (1 - 2 * b) * ssum(Wf[i][i] * r[i] for i in range(n)) + m * b * ssum(
            Wf[i][i] * p[i] for i in range(n)
        )
        return num / den

    obs = stat(c)
    sims = []
    cum = []
    acc = 0.0
    for v in p:
        acc += v
        cum.append(acc)
    for s in range(nsim):
        u = random_uniform(int(C), seed=seed, stream=s + 1)
        cc = [0.0] * n
        for uu in u:
            k = 0
            while k < n - 1 and float(uu) > cum[k]:
                k += 1
            cc[k] += 1.0
        sims.append(stat(cc))
    pv = (1 + sum(1 for v in sims if v >= obs)) / (nsim + 1) if nsim else float("nan")
    return RichResult(payload={"statistic": obs, "p_value": pv, "simulated": sims})


def _llr(c, e, C):
    if c <= e:
        return 0.0
    out = c * math.log(c / e)
    if C - c > 0:
        out += (C - c) * math.log((C - c) / (C - e))
    return out


def prospective_scan(
    counts, expected, coords, *, max_window: int = 3, max_pop_frac: float = 0.5, nsim: int = 99, seed: int = 0
) -> RichResult:
    r"""Kulldorff's prospective space-time scan statistic (Poisson model).

    ``counts[t][i]`` and ``expected[t][i]`` for times ``t`` (last is the
    present) and regions with centroids ``coords``. Candidate clusters are
    cylinders: circles around each centroid growing through the nearest
    regions until they exceed ``max_pop_frac`` of the total expected count,
    crossed with windows ending at the last time of length ``1..max_window``.
    The Poisson log-likelihood ratio
    ``c log(c/e) + (C - c) log((C - c)/(C - e))`` (``c > e``) is maximised;
    Monte Carlo replicates redistribute the total multinomially in proportion
    to the expected counts (Philox stream per replicate).

    References
    ----------
    Kulldorff, M. (2001). Prospective time periodic geographical disease
    surveillance using a scan statistic. JRSS A 164, 61-72. Kulldorff, M.
    (1997). A spatial scan statistic. Comm. Statist. Theory Methods 26, 1481-1496.

    Examples
    --------
    >>> cnt = [[1, 1, 1], [1, 1, 6]]
    >>> ex = [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]
    >>> r = prospective_scan(cnt, ex, [(0, 0), (1, 0), (5, 0)], max_window=1, nsim=19)
    >>> r.cluster, r.window
    ([2], 1)
    """
    T, n = len(counts), len(counts[0])
    Ctot = ssum(float(v) for row in counts for v in row)
    Etot = ssum(float(v) for row in expected for v in row)
    scale = Ctot / Etot
    E = [[float(v) * scale for v in row] for row in expected]
    zones = []
    area_e = [ssum(E[t][i] for t in range(T)) for i in range(n)]
    for i in range(n):
        order = sorted(
            range(n), key=lambda j: (math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]), j)
        )
        z, tot = [], 0.0
        for j in order:
            if tot + area_e[j] > max_pop_frac * Ctot and z:
                break
            z.append(j)
            tot += area_e[j]
            zones.append(sorted(z))
    uniq = []
    for z in zones:
        if z not in uniq:
            uniq.append(z)

    def best(Cmat):
        tot = ssum(v for row in Cmat for v in row)
        bl, bz, bw = 0.0, uniq[0], 1
        for z in uniq:
            for w in range(1, min(max_window, T) + 1):
                c = ssum(Cmat[t][i] for t in range(T - w, T) for i in z)
                e = ssum(E[t][i] for t in range(T - w, T) for i in z)
                v = _llr(c, e, tot)
                if v > bl + 1e-12:
                    bl, bz, bw = v, z, w
        return bl, bz, bw

    obs, zone, win = best([[float(v) for v in row] for row in counts])
    cells = [(t, i) for t in range(T) for i in range(n)]
    cum, acc = [], 0.0
    for t, i in cells:
        acc += E[t][i] / Ctot
        cum.append(acc)
    sims = []
    for s in range(nsim):
        u = random_uniform(int(Ctot), seed=seed, stream=s + 1)
        M = [[0.0] * n for _ in range(T)]
        for uu in u:
            k = 0
            while k < len(cells) - 1 and float(uu) > cum[k]:
                k += 1
            M[cells[k][0]][cells[k][1]] += 1.0
        sims.append(best(M)[0])
    pv = (1 + sum(1 for v in sims if v >= obs)) / (nsim + 1)
    return RichResult(payload={"llr": obs, "cluster": zone, "window": win, "p_value": pv})


def poisson_ecological(y, expected, X) -> RichResult:
    r"""Poisson ecological regression ``log E(y_i) = log E_i + x_i' beta`` by IRLS with a quasi-Poisson scale.

    Iteratively reweighted least squares on the working response; the
    dispersion ``phi = Pearson chi^2 / (n - p)``, standard errors scaled by
    ``sqrt(phi)`` (an intercept is added).

    References
    ----------
    Breslow, N. E. and Day, N. E. (1987). Statistical Methods in Cancer
    Research II, ch. 4. McCullagh, P. and Nelder, J. A. (1989). Generalized
    Linear Models, 2nd ed., section 6.2.

    Examples
    --------
    >>> r = poisson_ecological([2, 4, 8], [1.0, 1.0, 1.0], [[0.0], [1.0], [2.0]])
    >>> [round(v, 10) for v in r.beta]
    [0.6931471806, 0.6931471806]
    """
    ys = [float(v) for v in y]
    off = [math.log(float(v)) for v in expected]
    Xr = [[1.0] + [float(v) for v in r] for r in X]
    n, p = len(ys), len(Xr[0])
    beta = [math.log(max(ssum(ys), 0.5) / ssum(math.exp(o) for o in off))] + [0.0] * (p - 1)
    for _ in range(100):
        eta = [off[i] + ssum(Xr[i][k] * beta[k] for k in range(p)) for i in range(n)]
        mu = [math.exp(v) for v in eta]
        zw = [eta[i] - off[i] + (ys[i] - mu[i]) / mu[i] for i in range(n)]
        G = [[ssum(mu[i] * Xr[i][a] * Xr[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
        h = [ssum(mu[i] * Xr[i][a] * zw[i] for i in range(n)) for a in range(p)]
        new = solve(G, h)
        done = max(abs(a - c) for a, c in zip(new, beta)) <= 1e-12
        beta = new
        if done:
            break
    mu = [math.exp(off[i] + ssum(Xr[i][k] * beta[k] for k in range(p))) for i in range(n)]
    G = [[ssum(mu[i] * Xr[i][a] * Xr[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
    V = inverse(G)
    phi = ssum((ys[i] - mu[i]) ** 2 / mu[i] for i in range(n)) / (n - p) if n > p else float("nan")
    dev = 2 * ssum((ys[i] * math.log(ys[i] / mu[i]) if ys[i] > 0 else 0.0) - (ys[i] - mu[i]) for i in range(n))
    return RichResult(
        payload={
            "beta": beta,
            "se": [math.sqrt(V[k][k]) for k in range(p)],
            "se_quasi": [math.sqrt(phi * V[k][k]) for k in range(p)],
            "dispersion": phi,
            "deviance": dev,
            "fitted": mu,
        }
    )


def areal_wombling(values, A, *, quantile: float = 0.8) -> RichResult:
    r"""Areal wombling: boundary likelihood values ``|y_i - y_j|`` on adjacent pairs and the barrier edges.

    Edges whose boundary likelihood value is at or above its ``quantile``
    (type 7) are boundary (barrier) elements; returns all edges with values
    and the barrier set.

    References
    ----------
    Womble, W. H. (1951). Differential systematics. Science 114, 315-322. Lu,
    H. and Carlin, B. P. (2005). Bayesian areal wombling for geographical
    boundary analysis. Geographical Analysis 37, 265-285.

    Examples
    --------
    >>> r = areal_wombling([1.0, 1.2, 5.0], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], quantile=0.5)
    >>> r.barriers
    [[1, 2]]
    """
    y = [float(v) for v in values]
    n = len(y)
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if A[i][j]]
    blv = [abs(y[i] - y[j]) for i, j in edges]
    s = sorted(blv)
    h = (len(s) - 1) * quantile
    lo = int(math.floor(h))
    w = h - lo
    thr = (1 - w) * s[lo] + w * s[min(lo + 1, len(s) - 1)] if w > 0 else s[lo]
    return RichResult(
        payload={"edges": edges, "blv": blv, "threshold": thr, "barriers": [e for e, v in zip(edges, blv) if v >= thr]}
    )


def buffer_rate_ratio(cases, population, coords, sources, radius: float) -> RichResult:
    r"""Rate ratio of disease inside versus outside a buffer around point sources.

    Regions whose centroid lies within ``radius`` of any source form the
    exposed zone. ``RR = (c_in / n_in) / (c_out / n_out)`` with a Wald 95
    percent interval on the log scale (``se = sqrt(1/c_in + 1/c_out)``) and
    the conditional binomial test: given the total, ``c_in ~ Bin(C, n_in / N)``
    under no excess (one-sided upper p-value).

    References
    ----------
    Elliott, P., Wakefield, J. C., Best, N. G. and Briggs, D. J. (2000).
    Spatial Epidemiology, ch. 9. Diggle, P. J. (1990). A point process
    modelling approach to raised incidence of a rare phenomenon in the
    vicinity of a prespecified point. JRSS A 153, 349-362.

    Examples
    --------
    >>> r = buffer_rate_ratio([6, 2, 2], [100, 100, 100], [(0, 0), (5, 0), (9, 0)], [(0, 0)], 1.0)
    >>> r.rate_ratio
    3.0
    """
    ins = [any(math.hypot(x - sx, y - sy) <= radius for sx, sy in sources) for x, y in coords]
    ci = ssum(float(c) for c, f in zip(cases, ins) if f)
    co = ssum(float(c) for c, f in zip(cases, ins) if not f)
    ni = ssum(float(p) for p, f in zip(population, ins) if f)
    no = ssum(float(p) for p, f in zip(population, ins) if not f)
    rr = (ci / ni) / (co / no)
    se = math.sqrt(1 / ci + 1 / co) if ci > 0 and co > 0 else float("inf")
    C, q = int(ci + co), ni / (ni + no)
    lp = [
        math.lgamma(C + 1) - math.lgamma(k + 1) - math.lgamma(C - k + 1) + k * math.log(q) + (C - k) * math.log(1 - q)
        for k in range(int(ci), C + 1)
    ]
    pv = ssum(math.exp(v) for v in lp)
    return RichResult(
        payload={
            "rate_ratio": rr,
            "ci": [rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se)],
            "p_value": pv,
            "inside": ins,
        }
    )


def need_based_allocation(budget: float, population, need_index, *, floor_share: float = 0.0) -> list:
    r"""Weighted-capitation (need-based) resource allocation.

    Each area receives ``floor_share budget / n`` plus the remainder in
    proportion to its need-weighted population ``N_i x need_i`` (the
    capitation formula of the RAWP family, with need indices such as
    standardised mortality ratios).

    References
    ----------
    Department of Health and Social Security (1976). Sharing Resources for
    Health in England (RAWP report). Carr-Hill, R. A. et al. (1994). A
    formula for distributing NHS revenues based on small area use of
    hospital beds. University of York.

    Examples
    --------
    >>> need_based_allocation(100.0, [10, 20, 30], [1.0, 1.5, 0.5])
    [18.181818181818183, 54.54545454545455, 27.272727272727273]
    """
    n = len(population)
    wv = [float(p) * float(q) for p, q in zip(population, need_index)]
    tot = ssum(wv)
    base = floor_share * budget / n
    return [base + (1 - floor_share) * budget * v / tot for v in wv]


def funnel_control_limits(
    target: float, sizes, *, z=(1.959963984540054, 3.090232306167813), kind: str = "proportion", phi: float = 1.0
) -> RichResult:
    r"""Funnel-plot control limits for institutional or area indicators (Spiegelhalter 2005).

    ``proportion``: ``p0 +- z sqrt(phi p0 (1 - p0) / n)``; ``smr`` (ratio
    of observed to expected ``E``): ``1 +- z sqrt(phi / E)``; ``phi`` is an
    overdispersion factor (1 = none). Default ``z`` gives 95 and 99.8
    percent limits; returns lower and upper limits per ``z`` and size.

    References
    ----------
    Spiegelhalter, D. J. (2005). Funnel plots for comparing institutional
    performance. Statistics in Medicine 24, 1185-1202.

    Examples
    --------
    >>> r = funnel_control_limits(0.1, [100], z=(2.0,))
    >>> [round(v, 12) for v in r.lower[0] + r.upper[0]]
    [0.04, 0.16]
    """
    lower, upper = [], []
    for zz in z:
        lo, hi = [], []
        for nv in sizes:
            nv = float(nv)
            if kind == "smr":
                h = zz * math.sqrt(phi / nv)
                lo.append(max(1.0 - h, 0.0))
                hi.append(1.0 + h)
            else:
                h = zz * math.sqrt(phi * target * (1 - target) / nv)
                lo.append(max(target - h, 0.0))
                hi.append(min(target + h, 1.0))
        lower.append(lo)
        upper.append(hi)
    return RichResult(payload={"lower": lower, "upper": upper})


def ghose_drug_filter(mw, logp, mr, n_atoms) -> list:
    r"""Ghose-Viswanadhan-Wendoloski drug-likeness filter.

    A compound passes when ``160 <= MW <= 480``, ``-0.4 <= logP <= 5.6``,
    ``40 <= molar refractivity <= 130`` and ``20 <= atoms <= 70``
    (descriptors supplied by the caller).

    References
    ----------
    Ghose, A. K., Viswanadhan, V. N. and Wendoloski, J. J. (1999). A
    knowledge-based approach in designing combinatorial or medicinal
    chemistry libraries for drug discovery. J. Comb. Chem. 1, 55-68.

    Examples
    --------
    >>> ghose_drug_filter([300.0, 600.0], [2.0, 2.0], [80.0, 80.0], [35, 35])
    [True, False]
    """
    return [
        160 <= a <= 480 and -0.4 <= b <= 5.6 and 40 <= c <= 130 and 20 <= d <= 70
        for a, b, c, d in zip(mw, logp, mr, n_atoms)
    ]


def cheatsheet() -> str:
    return (
        "zip_gibbs / kernel_relative_risk / oden_ipop / prospective_scan / poisson_ecological / areal_wombling / "
        "buffer_rate_ratio / need_based_allocation / funnel_control_limits / ghose_drug_filter -> spatial epidemiology."
    )
