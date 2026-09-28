# morie.fn -- function file (rootcoder007/morie)
"""Compositional data analysis (Aitchison geometry, Dirichlet) and circular statistics (von Mises)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import qgamma

__all__ = [
    "aitchison_clr_covariance",
    "compositional_mad",
    "compositional_pielou",
    "compositional_quantile_dist",
    "aitchison_biplot",
    "dirichlet_fit_mom",
    "dirichlet_sample",
    "circular_summary",
    "vonmises_mle",
]


def _rows(X):
    A = np.asarray(X, dtype=float).tolist()
    return [[float(v) for v in (r if isinstance(r, list) else [r])] for r in A]


def _closure(x, total: float = 1.0):
    r"""Closure ``C(x) = total x / sum(x)`` of a composition (Aitchison 1986)."""
    v = [float(a) for a in x]
    if any(a <= 0 for a in v):
        raise ValueError("compositions must be strictly positive")
    s = ssum(v)
    return [total * a / s for a in v]


def _clr(x):
    r"""Centred log-ratio ``log(x_i / g(x))`` with ``g`` the geometric mean (Aitchison 1986)."""
    lv = [math.log(a) for a in _closure(x)]
    m = ssum(lv) / len(lv)
    return [a - m for a in lv]


def _aitchison_distance(x, y) -> float:
    r"""Aitchison distance: the Euclidean distance between the clr coordinates."""
    a, b = _clr(x), _clr(y)
    return math.sqrt(ssum((p - q) ** 2 for p, q in zip(a, b)))


def _median(v):
    t = sorted(v)
    k = len(t)
    return t[k // 2] if k % 2 else 0.5 * (t[k // 2 - 1] + t[k // 2])


def aitchison_clr_covariance(X) -> RichResult:
    r"""clr covariance ``cov(clr(X))`` with the variation matrix, total variance and centre (Aitchison 1986).

    ``clr_cov`` is the sample covariance (divisor ``n - 1``) of the clr
    coordinates and ``total_variance`` its trace (equal to the sum of the
    variation matrix ``var(log(x_i/x_j))`` over ``2D``); ``center`` is the
    closed geometric mean of each part.

    References
    ----------
    Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*.
    Chapman and Hall, London.

    Examples
    --------
    >>> round(aitchison_clr_covariance([[1, 2, 4], [2, 2, 2], [4, 2, 1]]).total_variance, 6)
    0.960906
    """
    R = _rows(X)
    n, D = len(R), len(R[0])
    L = [[math.log(a) for a in _closure(r)] for r in R]
    center = _closure([math.exp(ssum(L[i][j] for i in range(n)) / n) for j in range(D)])

    def var(v):
        m = ssum(v) / len(v)
        return ssum((a - m) ** 2 for a in v) / (len(v) - 1)

    T = [[var([L[i][a] - L[i][b] for i in range(n)]) for b in range(D)] for a in range(D)]
    C = [_clr(r) for r in R]
    mu = [ssum(C[i][j] for i in range(n)) / n for j in range(D)]
    cov = [
        [ssum((C[i][a] - mu[a]) * (C[i][b] - mu[b]) for i in range(n)) / (n - 1) for b in range(D)] for a in range(D)
    ]
    return RichResult(
        payload={"clr_cov": cov, "total_variance": ssum(cov[j][j] for j in range(D)), "variation": T, "center": center}
    )


def compositional_mad(X) -> list:
    r"""Median absolute deviation of each clr coordinate from its median, ``median(|z - median(z)|)``.

    Examples
    --------
    >>> [round(v, 6) for v in compositional_mad([[1, 2, 4], [2, 2, 2], [4, 2, 1]])]
    [0.693147, 0.0, 0.693147]
    """
    C = [_clr(r) for r in _rows(X)]
    out = []
    for j in range(len(C[0])):
        col = [r[j] for r in C]
        md = _median(col)
        out.append(_median([abs(a - md) for a in col]))
    return out


def compositional_pielou(x) -> float:
    r"""Pielou evenness ``J = H / log D`` of a composition, ``H = -sum p log p`` of its closure (Pielou 1966).

    References
    ----------
    Pielou, E. C. (1966). The measurement of diversity in different types of
    biological collections. *Journal of Theoretical Biology*, 13, 131-144.

    Examples
    --------
    >>> round(compositional_pielou([1, 1, 2]), 6)
    0.946395
    """
    p = _closure(x)
    return -ssum(a * math.log(a) for a in p) / math.log(len(p))


def compositional_quantile_dist(X) -> float:
    r"""Interquartile distance ``||clr(Q3) - clr(Q1)||`` of the part-wise quartiles (type 7, as ``quantile``).

    Examples
    --------
    >>> round(compositional_quantile_dist([[1, 2, 4], [2, 2, 2], [4, 2, 1], [3, 1, 1]]), 6)
    0.558805
    """
    R = _rows(X)
    n, D = len(R), len(R[0])

    def q(v, p):
        t = sorted(v)
        h = (len(t) - 1) * p
        lo = int(math.floor(h))
        return t[lo] + (h - lo) * (t[min(lo + 1, len(t) - 1)] - t[lo])

    q1 = [q([R[i][j] for i in range(n)], 0.25) for j in range(D)]
    q3 = [q([R[i][j] for i in range(n)], 0.75) for j in range(D)]
    return _aitchison_distance(q3, q1)


def aitchison_biplot(X) -> RichResult:
    r"""Compositional biplot factors: SVD ``U S V'`` of the column-centred clr matrix (Aitchison and Greenacre 2002).

    Returns the singular values, row scores ``U S`` and part loadings ``V``
    (signs fixed so the largest absolute loading of each axis is positive)
    and the proportion of total clr variance per axis.

    References
    ----------
    Aitchison, J. and Greenacre, M. (2002). Biplots of compositional data.
    *Journal of the Royal Statistical Society C*, 51(4), 375-392.

    Examples
    --------
    >>> b = aitchison_biplot([[1, 2, 4], [2, 2, 2], [4, 2, 1], [1, 1, 3]])
    >>> round(sum(b.explained), 12)
    1.0
    """
    R = _rows(X)
    C = [_clr(r) for r in R]
    n, D = len(C), len(C[0])
    mu = [ssum(C[i][j] for i in range(n)) / n for j in range(D)]
    Z = [[C[i][j] - mu[j] for j in range(D)] for i in range(n)]
    U, S, Vt = np.linalg.svd(np.asarray(Z, dtype=float), full_matrices=False)
    U, S, Vt = U.tolist(), [float(v) for v in S.tolist()], Vt.tolist()
    for k in range(len(S)):
        j = max(range(D), key=lambda t: abs(Vt[k][t]))
        if Vt[k][j] < 0:
            Vt[k] = [-v for v in Vt[k]]
            for i in range(n):
                U[i][k] = -U[i][k]
    tot = ssum(s * s for s in S)
    return RichResult(
        payload={
            "singular_values": S,
            "scores": [[U[i][k] * S[k] for k in range(len(S))] for i in range(n)],
            "loadings": [[Vt[k][j] for k in range(len(S))] for j in range(D)],
            "explained": [s * s / tot for s in S],
        }
    )


def dirichlet_fit_mom(X) -> RichResult:
    r"""Method-of-moments Dirichlet fit (Minka 2000): ``alpha_i = m_i s`` with ``s = m_1 (1 - m_1) / v_1 - 1``.

    ``m`` and ``v`` are the sample means and (divisor ``n - 1``) variances of
    the closed parts; the precision ``s`` uses the first part.

    References
    ----------
    Minka, T. P. (2000). Estimating a Dirichlet distribution. Technical
    report, MIT.

    Examples
    --------
    >>> [round(v, 6) for v in dirichlet_fit_mom([[1, 2, 7], [2, 2, 6], [3, 1, 6]]).alpha]
    [3.0, 2.5, 9.5]
    """
    R = [_closure(r) for r in _rows(X)]
    n, D = len(R), len(R[0])
    m = [ssum(R[i][j] for i in range(n)) / n for j in range(D)]
    v1 = ssum((R[i][0] - m[0]) ** 2 for i in range(n)) / (n - 1)
    s = m[0] * (1 - m[0]) / v1 - 1
    return RichResult(payload={"alpha": [a * s for a in m], "precision": s, "mean": m})


def dirichlet_sample(alpha, n: int, *, seed: int = 1) -> list:
    r"""Dirichlet draws ``C(g)`` with ``g_i ~ Gamma(alpha_i, 1)`` by inversion of Philox uniforms.

    Part ``i`` of draw ``k`` uses the quantile of uniform ``k`` on stream
    ``i`` of ``seed`` (identical in the R arm).

    Examples
    --------
    >>> [round(v, 6) for v in dirichlet_sample([2.0, 3.0], 1, seed=4)[0]]
    [0.058655, 0.941345]
    """
    a = [float(v) for v in alpha]
    U = [[float(v) for v in random_uniform(n, seed=seed, stream=i)] for i in range(len(a))]
    out = []
    for k in range(n):
        g = [qgamma(U[i][k], a[i], 1.0) for i in range(len(a))]
        s = ssum(g)
        out.append([v / s for v in g])
    return out


def circular_summary(theta) -> RichResult:
    r"""Circular mean direction, mean resultant length, circular variance and Rayleigh test (Mardia and Jupp 2000).

    ``C = mean cos``, ``S = mean sin``, mean direction ``atan2(S, C)`` in
    ``[0, 2 pi)``, ``Rbar = sqrt(C^2 + S^2)``, circular variance ``1 -
    Rbar``, circular standard deviation ``sqrt(-2 log Rbar)`` and the
    Rayleigh test p-value ``exp(-z)(1 + (2z - z^2)/(4n) - (24z - 132z^2 + 76z^3
    - 9z^4)/(288 n^2))`` with ``z = n Rbar^2`` (the second-order correction
    applied for ``n < 50``; Mardia and Jupp 2000, section 6.3.1; as
    ``circular::rayleigh.test``).

    References
    ----------
    Mardia, K. V. and Jupp, P. E. (2000). *Directional Statistics*. Wiley,
    Chichester.

    Examples
    --------
    >>> round(circular_summary([0.1, 0.3, 6.2]).rbar, 6)
    0.987794
    """
    t = [float(v) for v in theta]
    n = len(t)
    C = ssum(math.cos(v) for v in t) / n
    S = ssum(math.sin(v) for v in t) / n
    R = math.hypot(C, S)
    z = n * R * R
    p = math.exp(-z)
    if n < 50:
        p *= 1 + (2 * z - z * z) / (4 * n) - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n * n)
    return RichResult(
        payload={
            "mean": math.atan2(S, C) % (2 * math.pi),
            "rbar": R,
            "variance": 1 - R,
            "sd": math.sqrt(-2 * math.log(R)) if R > 0 else float("inf"),
            "rayleigh_p": min(max(p, 0.0), 1.0),
        }
    )


def _bessel_i(nu, x):
    """Modified Bessel function I_nu(x), nu in {0, 1}, by its power series."""
    t = (x / 2.0) ** nu / math.factorial(nu)
    s, k = t, 0
    while True:
        k += 1
        t *= (x / 2.0) ** 2 / (k * (k + nu))
        s += t
        if t < 1e-17 * s:
            return s


def _a1inv(r):
    """Inverse of A1(kappa) = I1(kappa)/I0(kappa) (Best and Fisher 1981), as circular::A1inv."""
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r**3 - 4 * r**2 + 3 * r)


def vonmises_mle(theta, *, bias: bool = False) -> RichResult:
    r"""Von Mises maximum likelihood: mean direction and concentration, as ``circular::mle.vonmises``.

    ``mu`` is the circular mean and ``kappa = A1^{-1}(mean cos(theta - mu))``
    by the Best and Fisher (1981) approximation (0 when that mean is not
    positive), standard errors ``1/sqrt(n kappa A1)`` and ``1/sqrt(n(1 -
    A1/kappa - A1^2))``; with ``bias`` the small-sample
    correction of Fisher (1993): ``max(kappa - 2/(n kappa), 0)`` if ``kappa <
    2``, else ``(n - 1)^3 kappa / (n^3 + n)``.  Also the log-likelihood
    ``kappa sum cos(theta - mu) - n log(2 pi I0(kappa))``.

    References
    ----------
    Best, D. J. and Fisher, N. I. (1981). The bias of the maximum likelihood
    estimators of the von Mises-Fisher concentration parameters.
    *Communications in Statistics - Simulation and Computation*, 10(5),
    493-502.

    Examples
    --------
    >>> round(vonmises_mle([0.1, 0.3, 6.2, 0.05]).kappa, 6)
    53.200971
    """
    t = [float(v) for v in theta]
    n = len(t)
    mu = math.atan2(ssum(math.sin(v) for v in t), ssum(math.cos(v) for v in t))
    V = ssum(math.cos(v - mu) for v in t) / n
    k = _a1inv(V) if V > 0 else 0.0
    if bias:
        k = max(k - 2 / (n * k), 0.0) if k < 2 else (n - 1) ** 3 * k / (n**3 + n)
    i0, i1 = _bessel_i(0, k), _bessel_i(1, k)
    a1 = i1 / i0
    ll = k * ssum(math.cos(v - mu) for v in t) - n * math.log(2 * math.pi * i0)
    se_mu = math.sqrt(1 / (n * k * a1)) if k > 0 else float("inf")
    se_k = math.sqrt(1 / (n * (1 - a1 / k - a1 * a1))) if k > 0 else float("nan")
    return RichResult(payload={"mu": mu % (2 * math.pi), "kappa": k, "se_mu": se_mu, "se_kappa": se_k, "loglik": ll})


def cheatsheet() -> str:
    return (
        "aitchison_clr_covariance / aitchison_biplot / dirichlet_fit_mom / vonmises_mle -> compositional and circular."
    )
