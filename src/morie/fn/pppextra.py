# morie.fn -- function file (rootcoder007/morie)
"""Point-pattern extras: k-th nearest-neighbour distances, the quadrat-count test of CSR, Morisita's
index of dispersion, the cross-type K and L functions, the Monte Carlo test of spatial segregation
of types, and Potts/Ising lattice models (Gibbs sampler and exact enumeration)."""

from __future__ import annotations

import itertools
import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import pchisq, qchisq
from .ripk import isotropic_weight

__all__ = [
    "knn_distances",
    "quadrat_test",
    "morisita_index",
    "cross_k",
    "segregation_test",
    "potts_gibbs",
    "potts_exact",
]


def _pts(P):
    return [(float(a), float(b)) for a, b in (P.tolist() if hasattr(P, "tolist") else P)]


def knn_distances(points, k: int = 1) -> RichResult:
    r"""Distance from each point to its ``k``-th nearest neighbour (as ``spatstat.geom::nndist(X, k = k)``), with its index and the mean.

    Ties are broken by the lower index. ``k = 2`` gives second-nearest
    neighbour distances (Thompson 1956).

    References
    ----------
    Thompson, H. R. (1956). Distribution of distance to nth neighbour in a
    population of randomly distributed individuals. *Ecology*, 37(2),
    391-394.

    Examples
    --------
    >>> knn_distances([(0, 0), (1, 0), (3, 0)], k=2).distance
    [3.0, 2.0, 3.0]
    """
    P = _pts(points)
    n = len(P)
    if not 1 <= k < n:
        raise ValueError("k must be between 1 and n - 1")
    dist, which = [], []
    for i in range(n):
        o = sorted((j for j in range(n) if j != i), key=lambda j: (math.dist(P[i], P[j]), j))
        dist.append(math.dist(P[i], P[o[k - 1]]))
        which.append(o[k - 1])
    return RichResult(payload={"distance": dist, "which": which, "mean": ssum(dist) / n, "k": k})


def quadrat_test(points, window, nx: int, ny: int) -> RichResult:
    r"""Pearson chi-square quadrat-count test of complete spatial randomness, as ``spatstat.explore::quadrat.test``.

    The rectangle ``window = (x0, x1, y0, y1)`` is cut into ``nx x ny``
    equal tiles (points on an inner boundary go to the upper tile); ``X^2 =
    sum (O - E)^2 / E`` with ``E = n / (nx ny)``, ``k - 1`` degrees of
    freedom. Returns the counts (rows from the top, as ``quadratcount``),
    the intensity ``n / |W|``, the upper-tail and the two-sided p-value
    (spatstat's default ``2 min(P_upper, P_lower)``).

    References
    ----------
    Greig-Smith, P. (1952). The use of random and contiguous quadrats in the
    study of the structure of plant communities. *Annals of Botany*, 16(2),
    293-316.

    Examples
    --------
    >>> r = quadrat_test([(0.1, 0.1), (0.2, 0.3), (0.7, 0.8), (0.9, 0.6)], (0, 1, 0, 1), 2, 2)
    >>> r.counts, r.statistic
    ([[0, 2], [2, 0]], 4.0)
    """
    P = _pts(points)
    x0, x1, y0, y1 = (float(v) for v in window)
    C = [[0] * nx for _ in range(ny)]
    for x, y in P:
        i = min(int((x - x0) / (x1 - x0) * nx), nx - 1)
        j = min(int((y - y0) / (y1 - y0) * ny), ny - 1)
        C[ny - 1 - j][i] += 1
    k = nx * ny
    E = len(P) / k
    X2 = ssum((c - E) ** 2 / E for row in C for c in row)
    up = float(pchisq(X2, k - 1, lower_tail=False))
    return RichResult(
        payload={
            "counts": C,
            "statistic": X2,
            "df": k - 1,
            "p_upper": up,
            "p_value": min(1.0, 2 * min(up, 1 - up)),
            "intensity": len(P) / ((x1 - x0) * (y1 - y0)),
        }
    )


def morisita_index(counts, *, crit: float = 0.05) -> RichResult:
    r"""Morisita's index of dispersion of quadrat counts and Smith-Gill's standardised index, as ``vegan::dispindmorisita``.

    ``I = n (sum x^2 - sum x) / ((sum x)^2 - sum x)`` over ``n`` quadrats (1
    random, > 1 clumped, < 1 uniform); uniform and clumped critical indices
    ``M_u``, ``M_c`` from the chi-square quantiles at ``crit / 2``; the
    standardised index ``I_st`` in ``[-1, 1]`` (Smith-Gill 1975; 0.5 and -0.5
    at the critical values) and the chi-square p-value of ``I (sum x - 1) + n
    - sum x`` on ``n - 1`` df.

    References
    ----------
    Morisita, M. (1959). Measuring of the dispersion of individuals and
    analysis of the distributional patterns. *Memoirs of the Faculty of
    Science, Kyushu University, Series E (Biology)*, 2, 215-235.
    Smith-Gill, S. J. (1975). Cytophysiological basis of disruptive pigmentary
    patterns in the leopard frog, Rana pipiens. *Journal of Morphology*,
    146(1), 35-54.

    Examples
    --------
    >>> round(morisita_index([0, 0, 0, 10]).imor, 12)
    4.0
    """
    x = [float(v) for v in counts]
    n = len(x)
    s, s2 = ssum(x), ssum(v * v for v in x)
    imor = n * (s2 - s) / (s * s - s)
    chi_lo = float(qchisq(1 - crit / 2, n - 1))
    chi_hi = float(qchisq(crit / 2, n - 1))
    muni = (chi_hi - n + s) / (s - 1)
    mclu = (chi_lo - n + s) / (s - 1)
    smor = imor
    if s > 1:
        if imor >= mclu > 1:
            smor = 0.5 + 0.5 * (imor - mclu) / (n - mclu)
        if mclu > imor >= 1:
            smor = 0.5 * (imor - 1) / (mclu - 1)
        if 1 > imor > muni:
            smor = -0.5 * (imor - 1) / (muni - 1)
        if 1 > muni > imor:
            smor = -0.5 + 0.5 * (imor - muni) / muni
    return RichResult(
        payload={
            "imor": imor,
            "mclu": mclu,
            "muni": muni,
            "imst": smor,
            "p_value": float(pchisq(imor * (s - 1) + n - s, n - 1, lower_tail=False)),
        }
    )


def cross_k(points, marks, i, j, window, r) -> RichResult:
    r"""Cross-type K and L functions ``K_ij(r) = |W| / (n_i n_j) sum_{a in i} sum_{b in j} e_ab 1(d_ab <= r)`` with Ripley's isotropic correction.

    ``e_ab`` is the reciprocal of the share of the circle about ``a`` through
    ``b`` inside the window. As ``spatstat.explore::Kcross(X, i, j, r,
    correction = "isotropic")`` on
    a rectangle ``(x0, x1, y0, y1)``; ``L = sqrt(K / pi)``. Under
    independence of the types ``K_ij(r) = pi r^2``.

    References
    ----------
    Lotwick, H. W. and Silverman, B. W. (1982). Methods for analysing spatial
    processes of several types of points. *JRSS B*, 44(3), 406-413.

    Examples
    --------
    >>> r = cross_k([(0.2, 0.2), (0.8, 0.8), (0.25, 0.2)], ["a", "a", "b"], "a", "b", (0, 1, 0, 1), [0.1])
    >>> round(r.K[0], 12)
    0.5
    """
    P = _pts(points)
    x0, x1, y0, y1 = (float(v) for v in window)
    pi_ = [p for p, m in zip(P, marks) if m == i]
    pj = [p for p, m in zip(P, marks) if m == j]
    area = (x1 - x0) * (y1 - y0)
    K = []
    for rr in r:
        s = 0.0
        for a in pi_:
            for b in pj:
                d = math.dist(a, b)
                if 0 < d <= rr:
                    s += 1.0 / isotropic_weight(a[0], a[1], d, x0, x1, y0, y1)
        K.append(area * s / (len(pi_) * len(pj)))
    return RichResult(
        payload={
            "r": [float(v) for v in r],
            "K": K,
            "L": [math.sqrt(v / math.pi) for v in K],
            "theo": [math.pi * float(v) ** 2 for v in r],
        }
    )


def segregation_test(points, marks, sigma: float, *, nsim: int = 19, seed: int = 1) -> RichResult:
    r"""Monte Carlo test of spatial segregation of types (Diggle, Zheng and Durr 2005), as ``spatstat.explore::segregation.test``.

    ``T = sum_i sum_m (p_m(x_i) - pbar_m)^2`` with leave-one-out Gaussian
    kernel (bandwidth ``sigma``) type probabilities ``p_m(x_i) = sum_{j != i,
    m_j = m} k_ij / sum_{j != i} k_ij`` (the uniform edge correction cancels)
    and ``pbar_m = n_m / n``; the p-value ``(1 + #{T_sim >= T}) / (1 + nsim)``
    from random relabellings (Philox Fisher-Yates, stream ``s`` for
    simulation ``s``).

    References
    ----------
    Diggle, P. J., Zheng, P. and Durr, P. (2005). Nonparametric estimation of
    spatial segregation in a multivariate point process: bovine
    tuberculosis in Cornwall, UK. *Applied Statistics*, 54(3), 645-658.

    Examples
    --------
    >>> r = segregation_test([(0, 0), (0.1, 0), (1, 1), (1.1, 1)], ["a", "a", "b", "b"], 0.2, nsim=5)
    >>> round(r.statistic, 6)
    2.0
    """
    P = _pts(points)
    n = len(P)
    types = sorted(set(marks))
    K = [
        [math.exp(-(math.dist(P[a], P[b]) ** 2) / (2 * sigma * sigma)) if a != b else 0.0 for b in range(n)]
        for a in range(n)
    ]
    den = [ssum(r) for r in K]

    def stat(lab):
        pbar = {m: sum(1 for v in lab if v == m) / n for m in types}
        return ssum(
            (ssum(K[a][b] for b in range(n) if lab[b] == m) / den[a] - pbar[m]) ** 2 for a in range(n) for m in types
        )

    obs = stat(list(marks))
    sims = []
    for s in range(nsim):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        lab = list(marks)
        for t in range(n - 1):
            k = t + int(u[t] * (n - t))
            lab[t], lab[k] = lab[k], lab[t]
        sims.append(stat(lab))
    return RichResult(
        payload={"statistic": obs, "simulated": sims, "p_value": (1 + sum(1 for v in sims if v >= obs)) / (1 + nsim)}
    )


def _neighbours(nr, nc):
    return [
        [(i + a) * nc + (j + b) for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= i + a < nr and 0 <= j + b < nc]
        for i in range(nr)
        for j in range(nc)
    ]


def potts_gibbs(
    nrow: int, ncol: int, q: int, beta: float, *, sweeps: int = 100, seed: int = 1, init=None
) -> RichResult:
    r"""Potts model on a free-boundary lattice by single-site Gibbs sampling (raster scan).

    ``P(x) propto exp(beta sum_{i~j} 1(x_i = x_j))`` over the four-neighbour
    pairs, states ``0..q-1``; each site is redrawn from its full conditional
    ``P(x_i = a | rest) propto exp(beta #{j ~ i: x_j = a})`` by inversion of a
    Philox uniform (stream = sweep). The Ising model with coupling ``J``
    (spins ``+-1``, ``exp(J sum s_i s_j)``) is ``q = 2``, ``beta = 2 J``.
    Returns the final lattice (rows), and per sweep the number of like
    neighbour pairs (the sufficient statistic) and the share of state 0.

    References
    ----------
    Potts, R. B. (1952). Some generalized order-disorder transformations.
    *Mathematical Proceedings of the Cambridge Philosophical Society*,
    48(1), 106-109.
    Geman, S. and Geman, D. (1984). Stochastic relaxation, Gibbs
    distributions, and the Bayesian restoration of images. *IEEE PAMI*,
    6(6), 721-741.

    Examples
    --------
    >>> r = potts_gibbs(3, 3, 2, 50.0, sweeps=5, init=[[0] * 3] * 3)
    >>> r.like_pairs[-1]
    12
    """
    nb = _neighbours(nrow, ncol)
    N = nrow * ncol
    x = (
        [v for row in init for v in row]
        if init is not None
        else [min(q - 1, int(u * q)) for u in random_uniform(N, seed=seed, stream=10**6)]
    )
    like, share = [], []
    for s in range(sweeps):
        u = [float(v) for v in random_uniform(N, seed=seed, stream=s)]
        for i in range(N):
            cnt = [0] * q
            for j in nb[i]:
                cnt[x[j]] += 1
            w = [math.exp(beta * (c - max(cnt))) for c in cnt]
            t = u[i] * ssum(w)
            a, acc = 0, w[0]
            while acc < t and a < q - 1:
                a += 1
                acc += w[a]
            x[i] = a
        like.append(sum(1 for i in range(N) for j in nb[i] if j > i and x[i] == x[j]))
        share.append(sum(1 for v in x if v == 0) / N)
    return RichResult(
        payload={
            "lattice": [x[r * ncol : (r + 1) * ncol] for r in range(nrow)],
            "like_pairs": like,
            "share_state0": share,
        }
    )


def potts_exact(nrow: int, ncol: int, q: int, beta: float) -> RichResult:
    r"""Exact Potts partition function and moments by enumeration of all ``q^(nrow ncol)`` states (small lattices).

    Returns ``log Z``, the mean and variance of the like-pair count ``S``
    (``d log Z / d beta`` and its derivative) and the exact full-conditional
    check value ``E[S]`` for validating samplers.

    Examples
    --------
    >>> r = potts_exact(1, 2, 2, 0.0)
    >>> round(math.exp(r.log_z), 12), r.mean_like_pairs
    (4.0, 0.5)
    """
    nb = _neighbours(nrow, ncol)
    N = nrow * ncol
    pairs = [(i, j) for i in range(N) for j in nb[i] if j > i]
    ws = []
    for st in itertools.product(range(q), repeat=N):
        ws.append(sum(1 for i, j in pairs if st[i] == st[j]))
    m = max(beta * s for s in ws)
    z = ssum(math.exp(beta * s - m) for s in ws)
    e1 = ssum(s * math.exp(beta * s - m) for s in ws) / z
    e2 = ssum(s * s * math.exp(beta * s - m) for s in ws) / z
    return RichResult(
        payload={"log_z": m + math.log(z), "mean_like_pairs": e1, "var_like_pairs": e2 - e1 * e1, "n_pairs": len(pairs)}
    )


def cheatsheet() -> str:
    return (
        "knn_distances / quadrat_test / morisita_index / cross_k / segregation_test / potts_gibbs / potts_exact -> "
        "point-pattern and lattice-model extras."
    )
