# morie.fn -- function file (rootcoder007/morie)
"""Health inequality indices, area deprivation scores, spatial accessibility (FCA family) and the radiation model."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "health_concentration_index",
    "spatial_gini",
    "theil_decomposition",
    "deprivation_index",
    "fca_accessibility",
    "gravity_accessibility",
    "nearest_facility",
    "radiation_flows",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).tolist()]


def _mat(M):
    return [[float(v) for v in r] for r in np.asarray(M, dtype=float).tolist()]


def health_concentration_index(health, rank_variable) -> RichResult:
    r"""Health concentration index ``C = 2 cov(h, r) / mean(h)`` with tied ranks averaged (Kakwani et al. 1997).

    ``r`` is the fractional rank of ``rank_variable`` (typically income):
    ``(rank - 1/2) / n`` with tied values given their mean rank; the
    covariance uses the divisor ``n``.  Negative values mean ``health`` is
    concentrated among the lower ranked.  Also returned: Erreygers' (2009)
    corrected index ``4 mean(h) C / (b - a)`` with ``[a, b]`` the sample
    range of ``h``.

    References
    ----------
    Kakwani, N., Wagstaff, A. and van Doorslaer, E. (1997). Socioeconomic
    inequalities in health: measurement, computation, and statistical
    inference. *Journal of Econometrics*, 77(1), 87-103.

    Examples
    --------
    >>> round(health_concentration_index([4.0, 3.0, 2.0, 1.0], [10, 20, 30, 40]).index, 6)
    -0.25
    """
    h, x = _vec(health), _vec(rank_variable)
    n = len(h)
    order = sorted(range(n), key=lambda i: x[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            rank[order[t]] = (r - 0.5) / n
        i = j + 1
    mh = ssum(h) / n
    mr = ssum(rank) / n
    cov = ssum((a - mh) * (b - mr) for a, b in zip(h, rank)) / n
    C = 2.0 * cov / mh
    lo, hi = min(h), max(h)
    return RichResult(
        payload={
            "index": C,
            "fractional_rank": rank,
            "erreygers": 4.0 * mh * C / (hi - lo) if hi > lo else float("nan"),
        }
    )


def spatial_gini(x, W=None) -> RichResult:
    r"""Gini coefficient with the Rey and Smith (2013) spatial decomposition.

    ``G = sum_ij |x_i - x_j| / (2 n^2 mean)`` (as ``ineq::Gini``), split into
    the neighbour part ``sum_ij w_ij |x_i - x_j| / (2 n^2 mean)`` and the
    non-neighbour part, ``w`` binary (any nonzero weight counts).

    References
    ----------
    Rey, S. J. and Smith, R. J. (2013). A spatial decomposition of the Gini
    coefficient. *Letters in Spatial and Resource Sciences*, 6(2), 55-70.

    Examples
    --------
    >>> round(spatial_gini([1.0, 2.0, 3.0, 4.0]).gini, 6)
    0.25
    """
    v = _vec(x)
    n = len(v)
    mu = ssum(v) / n
    den = 2.0 * n * n * mu
    tot = ssum(abs(a - b) for a in v for b in v) / den
    out = {"gini": tot}
    if W is not None:
        Wm = _mat(W)
        nbr = ssum(abs(v[i] - v[j]) for i in range(n) for j in range(n) if Wm[i][j] != 0) / den
        out.update({"neighbour": nbr, "non_neighbour": tot - nbr})
    return RichResult(payload=out)


def theil_decomposition(x, groups=None) -> RichResult:
    r"""Theil's T (Theil 1967), as ``ineq::Theil``, with the between/within group decomposition.

    ``T = (1/n) sum (x/mu) ln(x/mu)`` over positive values; with groups
    (Shorrocks 1980), ``T = sum_g s_g T_g + sum_g s_g ln(mu_g / mu)`` where
    ``s_g`` is group ``g``'s share of the total; also Theil's L (mean log
    deviation) ``mean ln(mu / x)``.

    References
    ----------
    Theil, H. (1967). *Economics and Information Theory*. North-Holland,
    Amsterdam.
    Shorrocks, A. F. (1980). The class of additively decomposable inequality
    measures. *Econometrica*, 48(3), 613-625.

    Examples
    --------
    >>> round(theil_decomposition([1.0, 2.0, 3.0, 4.0]).T, 6)
    0.10644
    """
    v = [a for a in _vec(x)]
    pos = [a for a in v if a != 0]
    mu = ssum(pos) / len(pos)
    T = ssum(a * math.log(a / mu) for a in pos) / ssum(pos)
    L = ssum(math.log(mu / a) for a in pos) / len(pos) if all(a > 0 for a in pos) else float("nan")
    out = {"T": T, "L": L}
    if groups is not None:
        g = list(groups)
        tot = ssum(v)
        mall = tot / len(v)
        within = between = 0.0
        comps = {}
        for k in sorted(set(g), key=lambda z: str(z)):
            xs = [a for a, b in zip(v, g) if b == k]
            sg = ssum(xs) / tot
            mg = ssum(xs) / len(xs)
            Tg = ssum(a * math.log(a / mg) for a in xs if a > 0) / ssum(xs)
            comps[k] = Tg
            within += sg * Tg
            between += sg * math.log(mg / mall)
        out.update({"within": within, "between": between, "group_T": comps})
    return RichResult(payload=out)


def deprivation_index(indicators, *, method: str = "townsend") -> RichResult:
    r"""Area deprivation scores as sums of standardised census indicators.

    ``townsend`` (Townsend, Phillimore and Beattie 1988): columns (percent)
    unemployment, households without a car, households not owner-occupied,
    overcrowding; unemployment and overcrowding are transformed ``ln(x +
    1)``, every column z-scored (mean, sample standard deviation) and
    summed.  ``carstairs`` (Carstairs and Morris 1991): overcrowding, male
    unemployment, no car, low social class, z-scored without transformation
    and summed.  ``sum`` z-scores and sums any columns.

    References
    ----------
    Townsend, P., Phillimore, P. and Beattie, A. (1988). *Health and
    Deprivation: Inequality and the North*. Croom Helm, London.
    Carstairs, V. and Morris, R. (1991). *Deprivation and Health in
    Scotland*. Aberdeen University Press.

    Examples
    --------
    >>> [round(v, 6) for v in deprivation_index([[1, 2], [3, 2], [5, 8]], method="sum").score]
    [-1.57735, -0.57735, 2.154701]
    """
    X = _mat(indicators)
    n, k = len(X), len(X[0])
    if method == "townsend":
        if k != 4:
            raise ValueError("townsend needs 4 columns: unemployment, no car, not owner-occupied, overcrowding")
        X = [[math.log(r[0] + 1.0), r[1], r[2], math.log(r[3] + 1.0)] for r in X]
    elif method == "carstairs":
        if k != 4:
            raise ValueError("carstairs needs 4 columns: overcrowding, male unemployment, no car, low social class")
    elif method != "sum":
        raise ValueError("method must be townsend, carstairs or sum")
    Z = [[0.0] * k for _ in range(n)]
    for c in range(k):
        col = [r[c] for r in X]
        m = ssum(col) / n
        sd = math.sqrt(ssum((a - m) ** 2 for a in col) / (n - 1))
        for i in range(n):
            Z[i][c] = (col[i] - m) / sd
    return RichResult(payload={"score": [ssum(r) for r in Z], "z": Z})


def fca_accessibility(
    supply, demand, D, d0: float, *, method: str = "2SFCA", steps=None, power: float = 2.0
) -> RichResult:
    r"""Floating catchment area accessibility of demand sites ``i`` to supply sites ``j``.

    With catchment weights ``W_ij`` (0 beyond ``d0``): ``R_j = S_j / sum_i
    W_ij P_i`` and ``A_i = sum_j W_ij R_j``.

    - ``2SFCA`` (Luo and Wang 2003): ``W = 1(d <= d0)``;
    - ``E2SFCA`` (Luo and Qi 2009): stepwise weights ``steps`` =
      ``[(upper distance, weight), ...]``;
    - ``KD2SFCA``: ``W = exp(-d^power)`` within ``d0`` (``SpatialAcc::ac``);
      ``gaussian`` the kernel of Dai (2010) ``(exp(-(d/d0)^2/2) -
      exp(-1/2)) / (1 - exp(-1/2))``;
    - ``3SFCA`` (Wan, Zou and Sternberg 2012): Gaussian weights with the
      selection probabilities ``G_ij = W_ij / sum_k W_ik``, ``R_j = S_j /
      sum_i G_ij W_ij P_i`` and ``A_i = sum_j G_ij W_ij R_j``.

    References
    ----------
    Luo, W. and Wang, F. (2003). Measures of spatial accessibility to health
    care in a GIS environment. *Environment and Planning B*, 30(6), 865-884.
    Luo, W. and Qi, Y. (2009). An enhanced two-step floating catchment area
    (E2SFCA) method for measuring spatial accessibility to primary care
    physicians. *Health and Place*, 15(4), 1100-1107.
    Wan, N., Zou, B. and Sternberg, T. (2012). A three-step floating
    catchment area method for analyzing spatial access to health services.
    *International Journal of Geographical Information Science*, 26(6),
    1073-1089.

    Examples
    --------
    >>> r = fca_accessibility([10, 5], [100, 200, 100], [[1, 5], [2, 2], [5, 1]], 3.0)
    >>> [round(v, 6) for v in r.access]
    [0.033333, 0.05, 0.016667]
    """
    S, P = _vec(supply), _vec(demand)
    Dm = _mat(D)
    n, m = len(P), len(S)
    if len(Dm) != n or len(Dm[0]) != m:
        raise ValueError("D must be demand x supply")

    def w(d):
        if d > d0:
            return 0.0
        if method == "2SFCA":
            return 1.0
        if method == "E2SFCA":
            for ub, wt in steps:
                if d <= ub:
                    return float(wt)
            return 0.0
        if method == "KD2SFCA":
            return math.exp(-(d**power))
        g = math.exp(-0.5 * (d / d0) ** 2)
        return (g - math.exp(-0.5)) / (1.0 - math.exp(-0.5))

    if method not in ("2SFCA", "E2SFCA", "KD2SFCA", "gaussian", "3SFCA"):
        raise ValueError("method must be 2SFCA, E2SFCA, KD2SFCA, gaussian or 3SFCA")
    if method == "E2SFCA" and not steps:
        raise ValueError("E2SFCA needs steps")
    W = [[w(Dm[i][j]) for j in range(m)] for i in range(n)]
    if method == "3SFCA":
        G = [[W[i][j] / ssum(W[i]) if ssum(W[i]) > 0 else 0.0 for j in range(m)] for i in range(n)]
    else:
        G = [[1.0] * m for _ in range(n)]
    R = []
    for j in range(m):
        den = ssum(G[i][j] * W[i][j] * P[i] for i in range(n))
        R.append(S[j] / den if den > 0 else 0.0)
    A = [ssum(G[i][j] * W[i][j] * R[j] for j in range(m)) for i in range(n)]
    return RichResult(payload={"access": A, "ratio": R})


def gravity_accessibility(supply, D, beta: float) -> list:
    r"""Hansen (1959) potential accessibility ``A_i = sum_j S_j exp(-beta d_ij)``, as ``SpatialAcc::ac(family = "Hansen")``.

    References
    ----------
    Hansen, W. G. (1959). How accessibility shapes land use. *Journal of the
    American Institute of Planners*, 25(2), 73-76.

    Examples
    --------
    >>> [round(v, 6) for v in gravity_accessibility([10, 5], [[1, 5], [2, 2]], 0.5)]
    [6.475732, 5.518192]
    """
    S = _vec(supply)
    return [ssum(s * math.exp(-beta * d) for s, d in zip(S, row)) for row in _mat(D)]


def nearest_facility(D) -> RichResult:
    r"""Distance to, and index of (0-based, first on ties), the nearest facility for each demand site.

    Examples
    --------
    >>> nearest_facility([[3, 1, 2], [0.5, 4, 4]]).index
    [1, 0]
    """
    Dm = _mat(D)
    idx = [min(range(len(r)), key=lambda j: (r[j], j)) for r in Dm]
    return RichResult(payload={"distance": [r[j] for r, j in zip(Dm, idx)], "index": idx})


def radiation_flows(population, coords, *, outflow=None) -> list:
    r"""Radiation model of mobility (Simini et al. 2012).

    ``T_ij = T_i m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))`` with ``m, n``
    the origin and destination populations and ``s_ij`` the population
    within distance ``d_ij`` of ``i`` (excluding ``i`` and ``j``); ``T_i``
    (default ``m_i``) is the total outflow of ``i``.

    References
    ----------
    Simini, F., Gonzalez, M. C., Maritan, A. and Barabasi, A.-L. (2012). A
    universal model for mobility and migration patterns. *Nature*,
    484(7392), 96-100.

    Examples
    --------
    >>> [round(v, 6) for v in radiation_flows([10, 20, 30], [(0, 0), (1, 0), (3, 0)])[0]]
    [0.0, 6.666667, 1.666667]
    """
    m = _vec(population)
    Pt = [tuple(float(v) for v in r) for r in np.asarray(coords, dtype=float).tolist()]
    n = len(m)
    T = m if outflow is None else _vec(outflow)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = math.dist(Pt[i], Pt[j])
            s = ssum(m[k] for k in range(n) if k not in (i, j) and math.dist(Pt[i], Pt[k]) < dij)
            out[i][j] = T[i] * m[i] * m[j] / ((m[i] + s) * (m[i] + m[j] + s))
    return out


def cheatsheet() -> str:
    return "health_concentration_index / spatial_gini / theil_decomposition / fca_accessibility -> health equity and access."
