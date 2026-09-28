# morie.fn -- function file (rootcoder007/morie)
"""Disease mapping: indirect standardisation, SMRs, empirical Bayes rate smoothing, probability and exceedance maps."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rrng_core import pgamma, qchisq, qgamma
from ._sci_core import gammaincc

__all__ = [
    "expected_counts",
    "standardized_ratio",
    "eb_global",
    "eb_local",
    "probability_map",
    "poisson_gamma_eb",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).tolist()]


def expected_counts(population, cases):
    r"""Expected counts by indirect standardisation, as ``SpatialEpi::expected``.

    ``population`` and ``cases`` are ``areas x strata`` tables; with the
    reference rates ``r_s = sum_i y_is / sum_i n_is`` pooled over the areas,
    ``E_i = sum_s n_is r_s``.

    Examples
    --------
    >>> expected_counts([[100, 50], [200, 50]], [[1, 2], [3, 3]])
    [3.8333333333333335, 5.166666666666667]
    """
    P = [_vec(r) for r in population]
    Y = [_vec(r) for r in cases]
    S = len(P[0])
    rate = [ssum(Y[i][s] for i in range(len(Y))) / ssum(P[i][s] for i in range(len(P))) for s in range(S)]
    return [ssum(P[i][s] * rate[s] for s in range(S)) for i in range(len(P))]


def standardized_ratio(observed, expected, *, conf: float = 0.95) -> RichResult:
    r"""Standardised incidence/mortality ratio ``y / E`` with exact Poisson limits (Garwood 1936).

    Lower ``chi^2_{alpha/2}(2y) / (2E)`` (0 when ``y = 0``), upper
    ``chi^2_{1-alpha/2}(2(y + 1)) / (2E)``.

    References
    ----------
    Garwood, F. (1936). Fiducial limits for the Poisson distribution.
    *Biometrika*, 28(3-4), 437-442.

    Examples
    --------
    >>> r = standardized_ratio([7], [5.5])
    >>> [round(v, 6) for v in (r.ratio[0], r.lower[0], r.upper[0])]
    [1.272727, 0.511702, 2.622305]
    """
    y, E = _vec(observed), _vec(expected)
    a = 1.0 - conf
    lo = [qchisq(a / 2, 2 * v) / (2 * e) if v > 0 else 0.0 for v, e in zip(y, E)]
    hi = [qchisq(1 - a / 2, 2 * (v + 1)) / (2 * e) for v, e in zip(y, E)]
    return RichResult(payload={"ratio": [v / e for v, e in zip(y, E)], "lower": lo, "upper": hi})


def eb_global(cases, population) -> RichResult:
    r"""Global empirical Bayes rate smoothing (Marshall 1991), as ``spdep::EBest`` (Poisson family).

    With raw rates ``p_i = y_i / n_i``, ``b = sum y / sum n``, ``s^2 = sum n_i
    (p_i - b)^2 / sum n``, the prior variance ``a = s^2 - b / nbar`` (0 if
    negative) and the estimate ``b + a (p_i - b) / (a + b / n_i)``.

    References
    ----------
    Marshall, R. J. (1991). Mapping disease and mortality rates using
    empirical Bayes estimators. *Journal of the Royal Statistical Society C*,
    40(2), 283-294.

    Examples
    --------
    >>> [round(v, 6) for v in eb_global([2, 10, 3], [100, 200, 150]).estimate]
    [0.033333, 0.033333, 0.033333]
    """
    y, n = _vec(cases), _vec(population)
    m = len(y)
    p = [a / b for a, b in zip(y, n)]
    ns = ssum(n)
    b = ssum(y) / ns
    s2 = ssum(ni * (pi - b) ** 2 for ni, pi in zip(n, p)) / ns
    a = max(0.0, s2 - b / (ns / m))
    est = [b + a * (pi - b) / (a + b / ni) for pi, ni in zip(p, n)]
    return RichResult(payload={"raw": p, "estimate": est, "a": a, "b": b})


def eb_local(cases, population, neighbours) -> RichResult:
    r"""Local empirical Bayes rate smoothing (Marshall 1991), as ``spdep::EBlocal``.

    Each unit's neighbourhood ``N_i`` includes itself; ``m_i = sum_{N_i} y /
    sum_{N_i} n``, ``C_i = sum_{j in N_i} n_j (x_j - m_j)^2`` (each term with
    the neighbour's own local mean, spdep's default), ``a_i = C_i / sum_{N_i}
    n - m_i / nbar_i`` (0 if negative, ``nbar_i`` the mean population of
    ``N_i``) and the estimate ``m_i + (x_i - m_i) a_i / (a_i + m_i / n_i)``.

    :param neighbours: list of neighbour index lists (0-based, self excluded).

    Examples
    --------
    >>> [round(v, 6) for v in eb_local([2, 10, 3], [100, 200, 150], [[1], [0, 2], [1]]).estimate]
    [0.037705, 0.039096, 0.033263]
    """
    y, n = _vec(cases), _vec(population)
    k = len(y)
    x = [a / b for a, b in zip(y, n)]
    Nb = [sorted(set([i] + list(nb))) for i, nb in enumerate(neighbours)]
    ri = [ssum(y[j] for j in N) for N in Nb]
    ni = [ssum(n[j] for j in N) for N in Nb]
    nbar = [ni[i] / len(Nb[i]) for i in range(k)]
    m = [ri[i] / ni[i] for i in range(k)]
    C = [ssum(n[j] * (x[j] - m[j]) ** 2 for j in N) for N in Nb]
    a = [max(0.0, C[i] / ni[i] - m[i] / nbar[i]) for i in range(k)]
    est = [m[i] + (x[i] - m[i]) * (a[i] / (a[i] + m[i] / n[i])) if a[i] + m[i] / n[i] > 0 else m[i] for i in range(k)]
    return RichResult(payload={"raw": x, "estimate": est, "a": a, "m": m})


def probability_map(cases, population, *, alternative: str = "less") -> RichResult:
    r"""Choynowski (1959) Poisson probability map, as ``spdep::probmap``.

    Expected counts ``n_i sum y / sum n``, relative risk ``100 y / E`` and
    ``P(Y <= y_i)`` (``less``) or ``P(Y >= y_i)`` (``greater``) under
    ``Poisson(E_i)``.

    References
    ----------
    Choynowski, M. (1959). Maps based on probabilities. *Journal of the
    American Statistical Association*, 54(286), 385-388.

    Examples
    --------
    >>> [round(v, 6) for v in probability_map([2, 10, 3], [100, 200, 150]).pmap]
    [0.352776, 0.923446, 0.265026]
    """
    if alternative not in ("less", "greater"):
        raise ValueError("alternative must be less or greater")
    y, n = _vec(cases), _vec(population)
    b = ssum(y) / ssum(n)
    E = [v * b for v in n]
    if alternative == "less":
        pm = [float(gammaincc(v + 1.0, e)) for v, e in zip(y, E)]
    else:
        pm = [1.0 if v <= 0 else 1.0 - float(gammaincc(v, e)) for v, e in zip(y, E)]
    return RichResult(
        payload={
            "raw": [a / c for a, c in zip(y, n)],
            "expected": E,
            "relrisk": [100.0 * a / e for a, e in zip(y, E)],
            "pmap": pm,
        }
    )


def _trigamma(x):
    """psi'(x) by recurrence to x >= 10 and the asymptotic series."""
    acc = 0.0
    while x < 10.0:
        acc += 1.0 / (x * x)
        x += 1.0
    x2 = 1.0 / (x * x)
    return acc + 1.0 / x + x2 / 2.0 + x2 / x * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 * (1.0 / 42.0 - x2 / 30.0)))


def _nb_fit(y, E, X):
    """Negative binomial log-linear ML fit with offset log E (theta, beta), Newton on (beta, log theta)."""
    from ._sci_core import digamma

    n, p = len(y), len(X[0])
    b = [math.log(ssum(y) / ssum(E))] + [0.0] * (p - 1)
    mu = [e * math.exp(ssum(X[i][c] * b[c] for c in range(p))) for i, e in enumerate(E)]
    m = ssum(mu) / n
    v = ssum((yi - mi) ** 2 for yi, mi in zip(y, mu)) / n
    th = m * m / (v - m) if v > m else 100.0
    lt = math.log(th)
    for _ in range(500):
        th = math.exp(lt)
        eta = [ssum(X[i][c] * b[c] for c in range(p)) for i in range(n)]
        mu = [E[i] * math.exp(eta[i]) for i in range(n)]
        # score and Hessian of the log-likelihood in (beta, log theta)
        gb = [ssum(X[i][c] * (y[i] - mu[i]) * th / (th + mu[i]) for i in range(n)) for c in range(p)]
        gt = ssum(
            float(digamma(y[i] + th))
            - float(digamma(th))
            + math.log(th / (th + mu[i]))
            + 1.0
            - (y[i] + th) / (th + mu[i])
            for i in range(n)
        )
        Hbb = [
            [
                -ssum(X[i][c] * X[i][d] * mu[i] * th * (y[i] + th) / (th + mu[i]) ** 2 for i in range(n))
                for d in range(p)
            ]
            for c in range(p)
        ]
        Hbt = [ssum(X[i][c] * mu[i] * (y[i] - mu[i]) / (th + mu[i]) ** 2 for i in range(n)) * th for c in range(p)]
        htt = ssum(
            _trigamma(y[i] + th) - _trigamma(th) + 1.0 / th - 2.0 / (th + mu[i]) + (y[i] + th) / (th + mu[i]) ** 2
            for i in range(n)
        )
        Htt = htt * th * th + gt * th
        g = gb + [gt * th]
        H = [Hbb[c] + [Hbt[c]] for c in range(p)] + [Hbt + [Htt]]
        Hi = [[float(v) for v in r] for r in inverse(H)]
        step = [ssum(Hi[a][c] * g[c] for c in range(p + 1)) for a in range(p + 1)]
        b = [b[c] - step[c] for c in range(p)]
        lt -= step[p]
        if max(abs(s) for s in step) < 1e-12:
            break
    th = math.exp(lt)
    mu = [E[i] * math.exp(ssum(X[i][c] * b[c] for c in range(p))) for i in range(n)]
    return b, th, mu


def poisson_gamma_eb(observed, expected, X=None, *, threshold: float | None = None) -> RichResult:
    r"""Poisson-gamma empirical Bayes relative risks (Clayton and Kaldor 1987), as ``SpatialEpi::eBayes``.

    ``y_i ~ Poisson(E_i theta_i)``, ``theta_i ~ Gamma(alpha, alpha / mu_i)``
    with ``log mu_i = x_i' beta``: ``(beta, alpha)`` are the negative
    binomial maximum likelihood estimates (``MASS::glm.nb``), the posterior
    is ``Gamma(alpha + y_i, (alpha + E_i mu_i) / mu_i)`` with mean ``RR_i = w_i
    SMR_i + (1 - w_i) mu_i``, ``w_i = E_i mu_i / (alpha + E_i mu_i)``; also the
    posterior median and, with ``threshold``, the exceedance probability
    ``P(theta_i > threshold)`` (``SpatialEpi::EBpostthresh``).

    References
    ----------
    Clayton, D. and Kaldor, J. (1987). Empirical Bayes estimates of
    age-standardized relative risks for use in disease mapping.
    *Biometrics*, 43(3), 671-681.

    Examples
    --------
    >>> r = poisson_gamma_eb([3, 8, 1, 12, 5, 2], [4.0, 5.5, 2.5, 7.0, 6.0, 3.0])
    >>> round(r.alpha, 4)
    587.2021
    """
    y, E = _vec(observed), _vec(expected)
    n = len(y)
    Xm = (
        [[1.0] for _ in range(n)]
        if X is None
        else [[1.0] + [float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]
    )
    beta, alpha, fitted = _nb_fit(y, E, Xm)
    mu = [f / e for f, e in zip(fitted, E)]
    w = [e * m / (alpha + e * m) for e, m in zip(E, mu)]
    smr = [a / e for a, e in zip(y, E)]
    rr = [wi * s + (1 - wi) * m for wi, s, m in zip(w, smr, mu)]
    med = [qgamma(0.5, alpha + a, (alpha + e * m) / m) for a, e, m in zip(y, E, mu)]
    out = {"RR": rr, "RRmed": med, "beta": beta, "alpha": alpha, "SMR": smr}
    if threshold is not None:
        out["exceedance"] = [
            pgamma(threshold, alpha + a, (alpha + e * m) / m, lower_tail=False) for a, e, m in zip(y, E, mu)
        ]
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "expected_counts / standardized_ratio / eb_global / eb_local / poisson_gamma_eb -> disease mapping."
