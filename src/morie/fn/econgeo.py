# morie.fn -- function file (rootcoder007/morie)
"""Regional economics: location quotients, Krugman/Hoover dissimilarity, LQ Gini, Herfindahl and concentration
ratios, Ellison-Glaeser agglomeration and coagglomeration, Duranton-Overman K-density."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = [
    "location_quotient",
    "krugman_index",
    "lq_gini",
    "herfindahl_index",
    "ellison_glaeser",
    "coagglomeration_index",
    "duranton_overman",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def location_quotient(E, *, method: str = "ratio"):
    r"""Location quotients of a regions x industries employment matrix ``E`` (``REAT::locq``).

    ``LQ_ri = (e_ri / e_r) / (e_i / e)`` with ``e_r`` the regional total,
    ``e_i`` the industry total and ``e`` the grand total; ``method="difference"``
    returns ``e_ri/e_r - e_i/e`` instead.

    References
    ----------
    Farhauer, O. and Kroell, A. (2014). *Standorttheorien*. Springer Gabler.
    Wieland, T. (2019). REAT: A regional economic analysis toolbox for R.
    *REGION*, 6(3), R1-R57.

    Examples
    --------
    >>> location_quotient([[10, 30], [30, 30]])
    [[0.625, 1.25], [1.25, 0.8333333333333334]]
    """
    M = [_vec(r) for r in E]
    tot = ssum(ssum(r) for r in M)
    ci = [ssum(M[r][i] for r in range(len(M))) / tot for i in range(len(M[0]))]
    out = []
    for r in M:
        er = ssum(r)
        out.append([(v / er) / c if method == "ratio" else v / er - c for v, c in zip(r, ci)])
    return out


def krugman_index(x, ref) -> RichResult:
    r"""Krugman dissimilarity ``K = sum |x_i/sum x - r_i/sum r|`` and the Hoover index ``K/2``.

    With ``x`` a region's industry employment and ``ref`` the national
    structure it is Krugman's specialization index (``REAT::krugman.spec``);
    with an industry's regional employment and total regional employment it
    is the concentration index (``REAT::krugman.conc``).

    References
    ----------
    Krugman, P. (1991). *Geography and Trade*. MIT Press.
    Hoover, E. M. (1936). The measurement of industrial localization.
    *Review of Economics and Statistics*, 18(4), 162-171.

    Examples
    --------
    >>> r = krugman_index([10, 30, 60], [30, 30, 40])
    >>> round(r.K, 6), round(r.hoover, 6)
    (0.4, 0.2)
    """
    a, b = _vec(x), _vec(ref)
    if len(a) != len(b):
        raise ValueError("x and ref must have equal length")
    sa, sb = ssum(a), ssum(b)
    K = ssum(abs(u / sa - v / sb) for u, v in zip(a, b))
    return RichResult(payload={"K": K, "hoover": K / 2})


def lq_gini(x, ref) -> float:
    r"""Gini coefficient of location quotients (``REAT::gini.spec`` / ``gini.conc``).

    With ``R_k = (x_k/sum x)/(ref_k/sum ref)`` sorted ascending,
    ``G = 2/(n^2 mean(R)) sum_k k (R_(k) - mean(R))``: 0 when ``x`` mirrors
    the reference structure.

    Examples
    --------
    >>> round(lq_gini([10, 30, 60], [30, 30, 40]), 6)
    0.27451
    """
    a, b = _vec(x), _vec(ref)
    sa, sb = ssum(a), ssum(b)
    R = sorted((u / sa) / (v / sb) for u, v in zip(a, b))
    n = len(R)
    m = ssum(R) / n
    return 2 / (n * n * m) * ssum((k + 1) * (r - m) for k, r in enumerate(R))


def herfindahl_index(x, *, k: int = 4) -> RichResult:
    r"""Herfindahl-Hirschman index ``H = sum s_i^2``, its normalised form ``(H - 1/n)/(1 - 1/n)``, ``1/H`` and ``CR_k``.

    Examples
    --------
    >>> r = herfindahl_index([50, 30, 20], k=2)
    >>> round(r.H, 6), round(r.normalized, 6), round(r.equivalent_number, 6), r.CR
    (0.38, 0.07, 2.631579, 0.8)
    """
    a = _vec(x)
    s = ssum(a)
    sh = [v / s for v in a]
    H = ssum(v * v for v in sh)
    n = len(a)
    return RichResult(
        payload={
            "H": H,
            "normalized": (H - 1 / n) / (1 - 1 / n) if n > 1 else float("nan"),
            "equivalent_number": 1 / H,
            "CR": ssum(sorted(sh, reverse=True)[:k]),
        }
    )


def _eg_parts(emp, region, regions, sj):
    ei = ssum(emp)
    sij = [ssum(e for e, g in zip(emp, region) if g == r) / ei for r in regions]
    G = ssum((u - v) ** 2 for u, v in zip(sij, sj))
    H = ssum((e / ei) ** 2 for e in emp)
    return sij, G, H


def ellison_glaeser(plant_emp, plant_region, region_emp=None) -> RichResult:
    r"""Ellison and Glaeser (1997) agglomeration index of one industry, as ``REAT::ellison.a``.

    ``gamma = (G - (1 - sum x_r^2) H) / ((1 - sum x_r^2)(1 - H))`` with the
    raw concentration ``G = sum_r (s_r - x_r)^2`` (industry share ``s_r`` and
    aggregate share ``x_r`` of region ``r``) and the plant Herfindahl ``H =
    sum_k z_k^2``. Also returned: the null z-statistic ``(G - (1 - sum
    x^2)H)/sqrt(Var G)`` with ``Var G = 2 [H^2 (sum x^2 - 2 sum x^3 + (sum
    x^2)^2) - sum z^4 (sum x^2 - 4 sum x^3 + 3 (sum x^2)^2)]``.
    ``region_emp`` gives aggregate employment by region (in the order of the
    sorted region labels; default the industry's own employment).

    References
    ----------
    Ellison, G. and Glaeser, E. L. (1997). Geographic concentration in U.S.
    manufacturing industries: a dartboard approach. *Journal of Political
    Economy*, 105(5), 889-927.

    Examples
    --------
    >>> r = ellison_glaeser([10, 20, 30, 40], ["a", "a", "b", "c"], [100, 100, 100])
    >>> round(r.gamma, 6), round(r.G, 6), round(r.H, 2)
    (-0.414286, 0.006667, 0.3)
    """
    emp = _vec(plant_emp)
    reg = list(plant_region)
    regions = sorted(set(reg), key=str)
    xe = _vec(region_emp) if region_emp is not None else [ssum(e for e, g in zip(emp, reg) if g == r) for r in regions]
    if len(xe) != len(regions):
        raise ValueError("region_emp needs one value per region")
    tx = ssum(xe)
    sj = [v / tx for v in xe]
    _, G, H = _eg_parts(emp, reg, regions, sj)
    s2 = ssum(v * v for v in sj)
    s3 = ssum(v**3 for v in sj)
    ei = ssum(emp)
    z4 = ssum((e / ei) ** 4 for e in emp)
    gamma = (G - (1 - s2) * H) / ((1 - s2) * (1 - H))
    varG = 2 * (H * H * (s2 - 2 * s3 + s2 * s2) - z4 * (s2 - 4 * s3 + 3 * s2 * s2))
    return RichResult(
        payload={"gamma": gamma, "G": G, "H": H, "z": (G - (1 - s2) * H) / math.sqrt(varG), "regions": regions}
    )


def coagglomeration_index(plant_emp, plant_industry, plant_region, region_emp) -> RichResult:
    r"""Ellison-Glaeser (1997, eq. 5) coagglomeration of a group of industries.

    ``gamma_c = [G/(1 - sum x_r^2) - H - sum_i gamma_i w_i^2 (1 - H_i)] / (1
    - sum_i w_i^2)`` with ``G`` the raw concentration of the group's pooled
    employment, ``w_i`` industry ``i``'s share of group employment, ``H_i`` and
    ``gamma_i`` its plant Herfindahl and :func:`ellison_glaeser` index, and
    ``H = sum_i w_i^2 H_i``. ``region_emp`` is aggregate employment by sorted
    region label.

    Examples
    --------
    >>> r = coagglomeration_index([10, 20, 30, 40, 25, 25], [1, 1, 1, 2, 2, 2], ["a", "b", "a", "a", "c", "b"],
    ...                           [100, 100, 100])
    >>> round(r.gamma_c, 6)
    0.083333
    """
    emp = _vec(plant_emp)
    ind, reg = list(plant_industry), list(plant_region)
    regions = sorted(set(reg), key=str)
    xe = _vec(region_emp)
    if len(xe) != len(regions):
        raise ValueError("region_emp needs one value per region")
    tx = ssum(xe)
    sj = [v / tx for v in xe]
    s2 = ssum(v * v for v in sj)
    _, G, _ = _eg_parts(emp, reg, regions, sj)
    tot = ssum(emp)
    inds = sorted(set(ind), key=str)
    w, Hs, gs = [], [], []
    for i in inds:
        e = [x for x, k in zip(emp, ind) if k == i]
        g = [r for r, k in zip(reg, ind) if k == i]
        _, Gi, Hi = _eg_parts(e, g, regions, sj)
        w.append(ssum(e) / tot)
        Hs.append(Hi)
        gs.append((Gi - (1 - s2) * Hi) / ((1 - s2) * (1 - Hi)))
    H = ssum(wi * wi * hi for wi, hi in zip(w, Hs))
    w2 = ssum(wi * wi for wi in w)
    gc = (G / (1 - s2) - H - ssum(gi * wi * wi * (1 - hi) for gi, wi, hi in zip(gs, w, Hs))) / (1 - w2)
    return RichResult(payload={"gamma_c": gc, "G": G, "H": H, "gamma_i": gs, "w": w, "industries": inds})


def _silverman(d):
    n = len(d)
    m = ssum(d) / n
    sd = math.sqrt(ssum((v - m) ** 2 for v in d) / (n - 1))
    s = sorted(d)
    iqr = _q7(s, 0.75) - _q7(s, 0.25)
    lo = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * lo * n ** (-0.2)


def _q7(s, p):
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _kd(P, r, bw):
    d = [math.dist(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    h = _silverman(d) if bw is None else float(bw)
    c = 1 / (len(d) * h * math.sqrt(2 * math.pi))
    return [
        c * ssum(math.exp(-0.5 * ((x - v) / h) ** 2) + math.exp(-0.5 * ((x + v) / h) ** 2) for v in d) for x in r
    ], h


def duranton_overman(
    coords, r, *, sites=None, n_sim: int = 0, bandwidth: float | None = None, level: float = 0.05, seed: int = 1
) -> RichResult:
    r"""Duranton and Overman (2005) K-density of bilateral distances with local counterfactual bands.

    ``K(d) = 1/(N h) sum_{i<j} [phi((d - d_ij)/h) + phi((d + d_ij)/h)]``
    over the ``N = n(n-1)/2`` pairwise distances, Gaussian kernel ``phi``,
    reflection at zero (Silverman 1986) and bandwidth ``h`` by Silverman's
    rule ``0.9 min(sd, IQR/1.34) N^{-1/5}`` unless given. With ``sites`` (all
    candidate locations, e.g. every establishment of the sector) and
    ``n_sim > 0``, the industry's ``n`` establishments are redrawn without
    replacement from ``sites`` (Philox uniforms, Fisher-Yates) and the local
    ``level/2`` and ``1 - level/2`` quantiles of the simulated densities give
    the confidence bands; localisation is ``K > upper`` (the global bands of
    DO 2005 are left to the user). The same bandwidth is used for every
    simulation.

    References
    ----------
    Duranton, G. and Overman, H. G. (2005). Testing for localization using
    micro-geographic data. *Review of Economic Studies*, 72(4), 1077-1106.

    Examples
    --------
    >>> k = duranton_overman([(0, 0), (1, 0), (0, 1)], [0.0, 1.0], bandwidth=0.5)
    >>> [round(v, 6) for v in k.K]
    [0.153718, 0.720813]
    """
    P = [tuple(float(v) for v in p) for p in coords]
    R = _vec(r)
    K, h = _kd(P, R, bandwidth)
    out = {"r": R, "K": K, "bandwidth": h}
    if sites is not None and n_sim > 0:
        S = [tuple(float(v) for v in p) for p in sites]
        n, m = len(P), len(S)
        sims = []
        for b in range(n_sim):
            u = [float(v) for v in random_uniform(n, seed=seed, stream=b)]
            idx = list(range(m))
            for t in range(n):
                j = t + int(u[t] * (m - t))
                idx[t], idx[j] = idx[j], idx[t]
            sims.append(_kd([S[i] for i in idx[:n]], R, h)[0])
        lo, hi = [], []
        for t in range(len(R)):
            col = sorted(s[t] for s in sims)
            lo.append(_q7(col, level / 2))
            hi.append(_q7(col, 1 - level / 2))
        out.update(
            {
                "lower": lo,
                "upper": hi,
                "localized": [k > u for k, u in zip(K, hi)],
                "dispersed": [k < v for k, v in zip(K, lo)],
            }
        )
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "location_quotient / krugman_index / ellison_glaeser / duranton_overman -> regional economics indices."
