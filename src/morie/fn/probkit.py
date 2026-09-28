# morie.fn -- function file (rootcoder007/morie)
"""Probability and inference toolkit: variance of a product, Weibull moments, the folded normal
and chi-square(1) laws, the harmonic-mean evidence estimator, CRPS of an arbitrary CDF, discrete
maximum entropy under moment constraints, bivariate logistic extreme-value simulation, the DDM
drift detector and KING kinship coefficients."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "product_variance",
    "weibull_moments",
    "folded_normal",
    "chisq1_cdf",
    "harmonic_mean_evidence",
    "crps_cdf",
    "max_entropy_discrete",
    "bv_logistic_simulate",
    "ddm_drift",
    "king_kinship",
]


def _pnorm(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def product_variance(mean_x: float, var_x: float, mean_y: float, var_y: float) -> float:
    r"""Variance of the product of independent random variables.

    ``Var(XY) = Var(X) Var(Y) + (EX)^2 Var(Y) + (EY)^2 Var(X)``
    (Blitzstein and Hwang 2019, ch. 10).

    References
    ----------
    Blitzstein, J. K. and Hwang, J. (2019). Introduction to Probability, 2nd
    ed. CRC Press, section 10.

    Examples
    --------
    >>> product_variance(1.0, 2.0, 3.0, 4.0)
    30.0
    """
    return var_x * var_y + mean_x**2 * var_y + mean_y**2 * var_x


def weibull_moments(lam: float, gamma: float) -> RichResult:
    r"""Mean and variance of ``T = X^(1/gamma)``, ``X ~ Expo(lam)`` (the Weibull ``Wei(lam, gamma)``).

    ``E(T) = Gamma(1 + 1/gamma) / lam^(1/gamma)`` and
    ``Var(T) = (Gamma(1 + 2/gamma) - Gamma(1 + 1/gamma)^2) / lam^(2/gamma)``
    (Blitzstein and Hwang 2019, Example 6.5.5 and exercise 10.35).

    Examples
    --------
    >>> r = weibull_moments(1.0, 1.0)
    >>> r.mean, r.var
    (1.0, 1.0)
    """
    m = math.gamma(1.0 + 1.0 / gamma) / lam ** (1.0 / gamma)
    v = (math.gamma(1.0 + 2.0 / gamma) - math.gamma(1.0 + 1.0 / gamma) ** 2) / lam ** (2.0 / gamma)
    return RichResult(payload={"mean": m, "var": v})


def folded_normal(y, mu: float = 0.0, sigma: float = 1.0) -> RichResult:
    r"""Folded normal law of ``|X|``, ``X ~ N(mu, sigma^2)`` (Leone, Nelson and Nottingham 1961).

    CDF ``Phi((y - mu)/sigma) + Phi((y + mu)/sigma) - 1`` (for ``mu = 0``,
    ``2 Phi(y/sigma) - 1``), density ``phi((y - mu)/sigma)/sigma +
    phi((y + mu)/sigma)/sigma`` for ``y >= 0``, mean
    ``sigma sqrt(2/pi) exp(-mu^2 / (2 sigma^2)) + mu (1 - 2 Phi(-mu/sigma))``
    and variance ``mu^2 + sigma^2 - mean^2``.

    References
    ----------
    Leone, F. C., Nelson, L. S. and Nottingham, R. B. (1961). The folded
    normal distribution. Technometrics 3, 543-550.

    Examples
    --------
    >>> r = folded_normal([1.0])
    >>> round(r.cdf[0], 12), round(r.mean, 12)
    (0.682689492137, 0.797884560803)
    """
    ys = [float(v) for v in (y if isinstance(y, (list, tuple)) else [y])]
    cdf, pdf = [], []
    for v in ys:
        if v < 0:
            cdf.append(0.0)
            pdf.append(0.0)
            continue
        a, b = (v - mu) / sigma, (v + mu) / sigma
        cdf.append(_pnorm(a) + _pnorm(b) - 1.0)
        pdf.append((math.exp(-a * a / 2) + math.exp(-b * b / 2)) / (sigma * math.sqrt(2 * math.pi)))
    mean = sigma * math.sqrt(2 / math.pi) * math.exp(-mu * mu / (2 * sigma * sigma)) + mu * (
        1 - 2 * _pnorm(-mu / sigma)
    )
    return RichResult(payload={"cdf": cdf, "pdf": pdf, "mean": mean, "var": mu * mu + sigma * sigma - mean * mean})


def chisq1_cdf(x) -> list:
    r"""CDF of ``Z^2``, ``Z ~ N(0, 1)``: ``P(-sqrt(x) <= Z <= sqrt(x)) = 2 Phi(sqrt(x)) - 1``.

    (Blitzstein and Hwang 2019, ch. 10, the chi-square with one degree of freedom.)

    Examples
    --------
    >>> [round(v, 12) for v in chisq1_cdf([0.0, 1.0])]
    [0.0, 0.682689492137]
    """
    xs = [float(v) for v in (x if isinstance(x, (list, tuple)) else [x])]
    return [2.0 * _pnorm(math.sqrt(v)) - 1.0 if v > 0 else 0.0 for v in xs]


def harmonic_mean_evidence(log_lik) -> RichResult:
    r"""Harmonic-mean estimator of the marginal likelihood (Newton and Raftery 1994).

    ``m(y) ~ ((1/S) sum_s 1 / p(y | theta_s))^(-1)`` from log-likelihoods at
    posterior draws, computed as ``log m = log S - logsumexp(-l_s)``. The
    estimator is consistent but has infinite variance in many models
    (Neal's comment to Newton and Raftery); report it with care.

    References
    ----------
    Newton, M. A. and Raftery, A. E. (1994). Approximate Bayesian inference
    with the weighted likelihood bootstrap. JRSS B 56, 3-48.

    Examples
    --------
    >>> round(harmonic_mean_evidence([-1.0, -2.0]).log_evidence, 12)
    -1.620114506958
    """
    ll = [float(v) for v in log_lik]
    m = max(-v for v in ll)
    lse = m + math.log(ssum(math.exp(-v - m) for v in ll))
    le = math.log(len(ll)) - lse
    return RichResult(payload={"log_evidence": le, "evidence": math.exp(le)})


def _exp_sinh(g, h=1.0 / 64.0, tmax=4.0):
    # int_0^inf g(u) du with u = exp(pi/2 sinh(tau)) (Takahasi-Mori exp-sinh rule)
    n = int(round(tmax / h))
    s = 0.0
    for k in range(-n, n + 1):
        t = k * h
        u = math.exp(math.pi / 2 * math.sinh(t))
        if u == 0.0 or math.isinf(u):
            continue
        s += g(u) * u * math.pi / 2 * math.cosh(t)
    return s * h


def _tanh_sinh(g, a, b, h=1.0 / 64.0, tmax=3.5):
    # int_a^b g(z) dz with z = (a + b)/2 + (b - a)/2 tanh(pi/2 sinh(tau)) (Takahasi-Mori tanh-sinh rule)
    n = int(round(tmax / h))
    c, r = (a + b) / 2.0, (b - a) / 2.0
    s = 0.0
    for k in range(-n, n + 1):
        t = k * h
        v = math.pi / 2 * math.sinh(t)
        x = math.tanh(v)
        if abs(x) >= 1.0:
            continue
        s += g(c + r * x) * math.pi / 2 * math.cosh(t) / math.cosh(v) ** 2
    return s * h * r


def crps_cdf(cdf, y: float, *, breaks=(), h: float = 1.0 / 64.0) -> float:
    r"""Continuous ranked probability score of a forecast CDF (Matheson and Winkler 1976).

    ``CRPS(F, y) = int (F(z) - 1{z >= y})^2 dz``. The line is cut at ``y`` and
    at the optional ``breaks`` (points where ``F`` is not smooth, such as the
    ends of a bounded support); the two tails are integrated by the exp-sinh
    and the finite pieces by the tanh-sinh double-exponential rules (step
    ``h``). ``cdf`` is any callable CDF.

    References
    ----------
    Matheson, J. E. and Winkler, R. L. (1976). Scoring rules for continuous
    probability distributions. Management Science 22, 1087-1096.
    Gneiting, T. and Raftery, A. E. (2007). JASA 102, 359-378.

    Examples
    --------
    >>> round(crps_cdf(lambda z: min(max(z, 0.0), 1.0), 0.5, breaks=(0.0, 1.0)), 12)
    0.083333333333
    """
    pts = sorted(set([float(y)] + [float(b) for b in breaks]))

    def g(z):
        return (cdf(z) - (1.0 if z >= y else 0.0)) ** 2

    total = _exp_sinh(lambda u: g(pts[0] - u), h) + _exp_sinh(lambda u: g(pts[-1] + u), h)
    for a, b in zip(pts[:-1], pts[1:]):
        total += _tanh_sinh(g, a, b, h)
    return total


def max_entropy_discrete(support_values, targets, *, prior=None, tol: float = 1e-12, max_iter: int = 200) -> RichResult:
    r"""Maximum-entropy distribution on a finite support under moment constraints (Jaynes 1957).

    ``support_values[k][i] = g_k(x_i)`` for constraint ``k``; the solution is
    ``p_i = q_i exp(sum_k lambda_k g_k(x_i)) / Z`` (``q`` a prior, uniform by
    default, giving minimum relative entropy), with ``lambda`` minimising the
    convex dual ``log Z(lambda) - lambda' m`` by damped Newton steps (a step
    is accepted on Armijo decrease or when it reduces the moment error).

    References
    ----------
    Jaynes, E. T. (1957). Information theory and statistical mechanics.
    Physical Review 106, 620-630.

    Examples
    --------
    >>> r = max_entropy_discrete([[1, 2, 3, 4, 5, 6]], [3.5])
    >>> [round(v, 12) for v in r.p]
    [0.166666666667, 0.166666666667, 0.166666666667, 0.166666666667, 0.166666666667, 0.166666666667]
    """
    G = [[float(v) for v in row] for row in support_values]
    m = [float(v) for v in targets]
    K, n = len(G), len(G[0])
    q = [1.0 / n] * n if prior is None else [float(v) for v in prior]
    lam = [0.0] * K

    def dual(lm):
        a = [math.log(q[i]) + ssum(lm[k] * G[k][i] for k in range(K)) for i in range(n)]
        mx = max(a)
        z = ssum(math.exp(v - mx) for v in a)
        p = [math.exp(v - mx) / z for v in a]
        return mx + math.log(z) - ssum(lm[k] * m[k] for k in range(K)), p

    f, p = dual(lam)
    it = 0
    for _ in range(max_iter):
        it += 1
        mu = [ssum(p[i] * G[k][i] for i in range(n)) for k in range(K)]
        g = [mu[k] - m[k] for k in range(K)]
        if max(abs(v) for v in g) <= tol:
            break
        H = [[ssum(p[i] * (G[a][i] - mu[a]) * (G[b][i] - mu[b]) for i in range(n)) for b in range(K)] for a in range(K)]
        d = solve(H, [-v for v in g])
        step = 1.0
        gd = ssum(g[k] * d[k] for k in range(K))
        gmax = max(abs(v) for v in g)
        while True:
            new = [lam[k] + step * d[k] for k in range(K)]
            fn, pn = dual(new)
            gn = max(abs(ssum(pn[i] * G[k][i] for i in range(n)) - m[k]) for k in range(K))
            if fn <= f + 1e-4 * step * gd or gn < gmax or step < 1e-12:
                break
            step /= 2.0
        lam, f, p = new, fn, pn
    ent = -ssum(v * math.log(v) for v in p if v > 0)
    return RichResult(payload={"p": p, "lambdas": lam, "entropy": ent, "iterations": it})


def bv_logistic_simulate(
    n: int, alpha: float, *, loc=(0.0, 0.0), scale=(1.0, 1.0), shape=(0.0, 0.0), seed: int = 0
) -> RichResult:
    r"""Simulate the bivariate logistic extreme-value distribution by Shi's (1995) mixture method.

    With ``U ~ U(0,1)``, ``Z ~ Gamma(2, 1)`` with probability ``alpha`` and
    ``Expo(1)`` otherwise, ``(1 / (Z U^alpha), 1 / (Z (1 - U)^alpha))`` has
    unit Frechet margins and logistic dependence ``alpha`` (1 = independence);
    margins are then mapped to GEV(``loc, scale, shape``). This is the
    algorithm of ``evd::rbvevd(model = "log")`` (Stephenson 2003); draws come
    from Philox streams 0-3 of ``seed``.

    References
    ----------
    Shi, D. (1995). Fisher information for a multivariate extreme value
    distribution. Biometrika 82, 644-649. Stephenson, A. (2003). Simulating
    multivariate extreme value distributions of logistic type. Extremes 6, 49-59.

    Examples
    --------
    >>> r = bv_logistic_simulate(3, 0.5, seed=1)
    >>> len(r.x), len(r.y)
    (3, 3)
    """
    u = random_uniform(n, seed=seed, stream=0)
    mix = random_uniform(n, seed=seed, stream=1)
    e1 = random_uniform(n, seed=seed, stream=2)
    e2 = random_uniform(n, seed=seed, stream=3)
    xs, ys = [], []
    for i in range(n):
        z = -math.log(float(e1[i]))
        if float(mix[i]) < alpha:
            z += -math.log(float(e2[i]))
        f1 = 1.0 / (z * float(u[i]) ** alpha)
        f2 = 1.0 / (z * (1.0 - float(u[i])) ** alpha)
        out = []
        for f, lo, sc, sh in ((f1, loc[0], scale[0], shape[0]), (f2, loc[1], scale[1], shape[1])):
            out.append(lo + sc * math.log(f) if sh == 0 else lo + sc * (f**sh - 1.0) / sh)
        xs.append(out[0])
        ys.append(out[1])
    return RichResult(payload={"x": xs, "y": ys})


def ddm_drift(errors, *, min_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0) -> RichResult:
    r"""Drift Detection Method (Gama et al. 2004) on a stream of 0/1 prediction errors.

    ``p_t`` is the running error rate and ``s_t = sqrt(p_t (1 - p_t) / t)``;
    after ``min_instances`` samples the minimum of ``p + s`` is tracked, a
    warning (``warning_points``) is raised when ``p_t + s_t > p_min + warning_level s_min`` and a
    drift when ``p_t + s_t > p_min + drift_level s_min``, after which the
    statistics restart.

    References
    ----------
    Gama, J., Medas, P., Castillo, G. and Rodrigues, P. (2004). Learning
    with drift detection. SBIA 2004, LNCS 3171, 286-295.

    Examples
    --------
    >>> r = ddm_drift([0] * 40 + [1] * 20)
    >>> r.drifts
    [40]
    """
    warns, drifts = [], []
    t, p = 0, 1.0
    s = 0.0
    pmin = smin = psmin = math.inf
    for i, e in enumerate(errors):
        t += 1
        p = p + (float(e) - p) / t
        s = math.sqrt(p * (1 - p) / t)
        if t < min_instances:
            continue
        if p + s <= psmin:
            pmin, smin, psmin = p, s, p + s
        if p + s > pmin + drift_level * smin:
            drifts.append(i)
            t, p, s = 0, 1.0, 0.0
            pmin = smin = psmin = math.inf
        elif p + s > pmin + warning_level * smin:
            warns.append(i)
    return RichResult(payload={"warning_points": warns, "drifts": drifts})


def king_kinship(genotypes, *, method: str = "robust") -> RichResult:
    r"""KING kinship coefficients from genotype counts (Manichaikul et al. 2010).

    ``genotypes[i][s]`` is the minor-allele count 0/1/2 (None or NaN
    missing). Over SNPs typed in both individuals, with ``N_Aa`` het counts,
    ``N_Aa,Aa`` shared hets and ``N_AA,aa`` opposite homozygotes:
    robust between-family ``phi = (N_Aa,Aa - 2 N_AA,aa) / (2 N_i) + 1/2 -
    (N_i + N_j) / (4 N_i)`` with ``N_i = min(N_Aa^(i), N_Aa^(j))``;
    within-family (``method="within"``)
    ``phi = (N_Aa,Aa - 2 N_AA,aa) / (N_Aa^(i) + N_Aa^(j))``; NaN when the
    denominator has no heterozygotes.

    References
    ----------
    Manichaikul, A. et al. (2010). Robust relationship inference in
    genome-wide association studies. Bioinformatics 26, 2867-2873.

    Examples
    --------
    >>> r = king_kinship([[0, 1, 2, 1], [0, 1, 2, 1]])
    >>> r.kinship[0][1]
    0.5
    """
    n = len(genotypes)
    K = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            hi = hj = hh = oo = 0
            for a, b in zip(genotypes[i], genotypes[j]):
                if a is None or b is None or a != a or b != b:
                    continue
                hi += a == 1
                hj += b == 1
                hh += a == 1 and b == 1
                oo += (a == 0 and b == 2) or (a == 2 and b == 0)
            nm = min(hi, hj)
            if hi + hj == 0 or (method != "within" and nm == 0):
                phi = float("nan")
            elif method == "within":
                phi = (hh - 2 * oo) / (hi + hj)
            else:
                phi = (hh - 2 * oo) / (2 * nm) + 0.5 - (hi + hj) / (4 * nm)
            K[i][j] = K[j][i] = phi
    return RichResult(payload={"kinship": K})


def cheatsheet() -> str:
    return (
        "product_variance / weibull_moments / folded_normal / chisq1_cdf / harmonic_mean_evidence / crps_cdf / "
        "max_entropy_discrete / bv_logistic_simulate / ddm_drift / king_kinship -> probability and inference toolkit."
    )
