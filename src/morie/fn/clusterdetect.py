# morie.fn -- function file (rootcoder007/morie)
"""Disease and crime cluster detection beyond the circular scan: elliptic scan (Kulldorff et al. 2006),
flexibly shaped scan FleXScan (Tango and Takahashi 2005), normal-model scan for continuous outcomes
(Kulldorff, Huang and Konty 2009), Cuzick-Edwards k-nearest-neighbour case-control test, the focused
Lawson-Waller score test, and fixed-circle Openshaw GAM / Rushton-Lolonis moving-window scans."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import pnorm
from ._sci_core import gammaincc
from .scanstat import _llr_binom, _multinomial, _nn, _pts, _vec

__all__ = [
    "elliptic_scan",
    "flex_zones",
    "flexscan",
    "normal_scan",
    "cuzick_edwards",
    "lawson_waller",
    "fixed_circle_scan",
]


def _xlr(y, e):
    return y * (math.log(y) - math.log(e)) if y > 0 else 0.0


def _stat_poisson(yin, ty, ein, eout, min_cases):
    """smerc Poisson LLR: zero unless yin >= min_cases and the zone rate exceeds the outside rate."""
    if yin < min_cases or yin <= 0:
        return 0.0
    yout = ty - yin
    lrin = math.log(yin) - math.log(ein)
    lrout = (math.log(yout) - math.log(eout)) if yout > 0 else -math.inf
    if lrin <= lrout:
        return 0.0
    return yin * lrin + (yout * lrout if yout > 0 else 0.0)


def _pvals(tobs, tmax):
    n = len(tmax)
    return [(1 + sum(1 for t in tmax if t >= x)) / (n + 1) for x in tobs] if n else [1.0] * len(tobs)


def _prune(zones, tobs, pv, alpha):
    sig = [i for i, p in enumerate(pv) if p <= alpha] or [min(range(len(pv)), key=lambda i: pv[i])]
    return sig


def _ell_nn(P, pop, ubpop, shapes, nangles):
    tp = ssum(pop)
    nn, shp, ang = [], [], []
    for s, na in zip(shapes, nangles):
        angles = [90 + 180 * t / na for t in range(na)]
        for a in angles:
            rad = a * math.pi / 180
            sr, cr = math.sin(rad), math.cos(rad)
            for i in range(len(P)):
                d = []
                for j in range(len(P)):
                    dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
                    m1 = (dx * cr + dy * sr) / s
                    m2 = dx * sr - dy * cr
                    d.append(math.sqrt(m1 * m1 + m2 * m2))
                order = sorted(range(len(P)), key=lambda j: (d[j], j))
                cs, keep = 0.0, []
                for j in order:
                    cs += pop[j]
                    if cs <= tp * ubpop:
                        keep.append(j)
                    else:
                        break
                nn.append(keep)
                shp.append(float(s))
                ang.append(a)
    return nn, shp, ang


def _ell_stats(nn, y, ex, ty, pen, a, min_cases):
    out = []
    for lst, p in zip(nn, pen):
        yin = ein = 0.0
        row = []
        for j in lst:
            yin += y[j]
            ein += ex[j]
            t = _stat_poisson(yin, ty, ein, ty - ein, min_cases)
            row.append(t * p if a > 0 and p < 1 else t)
        out.append(row)
    return out


def _noc_start(nn, stats):
    """smerc ``noc_enn``: greedy non-overlapping clusters; lists are dropped by their starting region."""
    stats = [list(r) for r in stats]
    start = [lst[0] if lst else None for lst in nn]
    remaining = [i for i in range(len(nn)) if nn[i]]
    clusts, tobs, which = [], [], []
    cur = 1.0
    while remaining and cur > 0:
        best_i, best_t = None, -math.inf
        for i in remaining:
            m = max(stats[i]) if stats[i] else -math.inf
            if m > best_t:
                best_i, best_t = i, m
        if best_i is None or best_t == -math.inf:
            break
        k = max(range(len(stats[best_i])), key=lambda t: (stats[best_i][t], -t))
        clusts.append(nn[best_i][: k + 1])
        which.append(best_i)
        cur = best_t
        tobs.append(best_t)
        used = {j for c in clusts for j in c}
        remaining = [i for i in range(len(nn)) if nn[i] and start[i] not in used]
        for i in remaining:
            for t, j in enumerate(nn[i]):
                if j in used:
                    stats[i] = stats[i][:t]
                    break
    return clusts, tobs, which


def elliptic_scan(
    coords,
    cases,
    pop,
    *,
    ex=None,
    ubpop: float = 0.5,
    shape=(1, 1.5, 2, 3, 4, 5),
    nangle=(1, 4, 6, 9, 12, 15),
    a: float = 0.5,
    nsim: int = 499,
    alpha: float = 0.1,
    min_cases: int = 2,
    seed: int = 1,
) -> RichResult:
    r"""Kulldorff's elliptic spatial scan statistic with the eccentricity penalty, as ``smerc::elliptic.test``.

    Candidate zones are nearest-neighbour prefixes (cumulative population at
    most ``ubpop`` of the total) in the elliptic distance ``sqrt(((dx cos t +
    dy sin t) / s)^2 + (dx sin t - dy cos t)^2)`` for each shape ``s`` with
    ``nangle`` angles ``t`` in ``[90, 270)`` degrees, every centroid. The
    Poisson LLR ``y ln(y/E) + (Y - y) ln((Y - y)/(Y - E))`` (zones with more
    than expected cases and at least ``min_cases``) is multiplied by the
    penalty ``(4 s / (s + 1)^2)^a``. Non-overlapping clusters are chosen
    greedily; p-values use the maxima of ``nsim`` multinomial redistributions
    of the cases proportional to ``ex`` (Philox streams ``1..nsim``).

    References
    ----------
    Kulldorff, M., Huang, L., Pickle, L. and Duczmal, L. (2006). An elliptic
    spatial scan statistic. *Statistics in Medicine*, 25(22), 3929-3943.

    Examples
    --------
    >>> P = [(float(i), 0.0) for i in range(6)] + [(float(i), 1.0) for i in range(6)]
    >>> y = [9, 8, 9, 8, 0, 0, 0, 1, 0, 0, 1, 0]
    >>> r = elliptic_scan(P, y, [10] * 12, nsim=0)
    >>> sorted(r.clusters[0]["zone"]), r.clusters[0]["shape"], r.clusters[0]["angle"]
    ([0, 1, 2, 3], 2.0, 180.0)
    """
    P = _pts(coords)
    y, pp = _vec(cases), _vec(pop)
    ty = ssum(y)
    e = [v * ty / ssum(pp) for v in pp] if ex is None else _vec(ex)
    nn, shp, ang = _ell_nn(P, pp, ubpop, list(shape), list(nangle))
    pen = [(4 * s / (s + 1) ** 2) ** a for s in shp]
    zones, tobs, which = _noc_start(nn, _ell_stats(nn, y, e, ty, pen, a, min_cases))
    tmax = []
    for s in range(1, nsim + 1):
        ys = [float(v) for v in _multinomial(ty, e, seed, s)]
        tmax.append(max((v for r in _ell_stats(nn, ys, e, ty, pen, a, min_cases) for v in r), default=0.0))
    pv = _pvals(tobs, tmax)
    clusters = [
        {
            "zone": zones[i],
            "llr": tobs[i],
            "pvalue": pv[i],
            "shape": shp[which[i]],
            "angle": ang[which[i]],
            "cases": ssum(y[j] for j in zones[i]),
            "expected": ssum(e[j] for j in zones[i]),
        }
        for i in _prune(zones, tobs, pv, alpha)
    ]
    return RichResult(
        payload={
            "clusters": clusters,
            "all_zones": zones,
            "all_tobs": tobs,
            "all_pvalues": pv,
            "all_shapes": [shp[i] for i in which],
            "all_angles": [ang[i] for i in which],
        }
    )


def flex_zones(coords, W, k: int = 10):
    """FleXScan candidate zones: connected (by ``W``) subsets of each region's ``k`` nearest neighbours containing it."""
    P = _pts(coords)
    n = len(P)
    A = [[float(v) != 0 for v in row] for row in (W.tolist() if hasattr(W, "tolist") else W)]
    out, seen = [], set()
    for i in range(n):
        knn = sorted(range(n), key=lambda j: (math.dist(P[i], P[j]), j))[:k]
        level = [frozenset([knn[0]])]
        for z in level:
            if z not in seen:
                seen.add(z)
                out.append(sorted(z))
        for _ in range(k - 1):
            nxt, lev_seen = [], set()
            for z in level:
                for v in knn:
                    if v not in z and any(A[v][u] for u in z):
                        z2 = z | {v}
                        if z2 not in lev_seen:
                            lev_seen.add(z2)
                            nxt.append(z2)
            if not nxt:
                break
            for z in nxt:
                if z not in seen:
                    seen.add(z)
                    out.append(sorted(z))
            level = nxt
    return out


def flexscan(
    coords,
    cases,
    pop,
    W,
    *,
    k: int = 10,
    ex=None,
    kind: str = "poisson",
    nsim: int = 499,
    alpha: float = 0.1,
    seed: int = 1,
) -> RichResult:
    r"""Tango and Takahashi's flexibly shaped spatial scan statistic, as ``smerc::flex.test``.

    Zones (:func:`flex_zones`) are the subsets of each region's ``k``
    nearest regions (itself first) that contain it and are connected in the
    adjacency matrix ``W``. Each zone gets the Poisson or binomial likelihood
    ratio (zero unless the zone rate exceeds the outside rate); clusters are
    the significant zones (``pvalue <= alpha``, else the most likely one)
    taken in decreasing order of the statistic without overlap. P-values use
    the maxima over ``nsim`` multinomial redistributions (Philox). When no zone is
    significant the most likely zone is returned (smerc returns the first
    zone of smallest p-value, which with tied p-values need not be it).

    References
    ----------
    Tango, T. and Takahashi, K. (2005). A flexibly shaped spatial scan
    statistic for detecting clusters. *International Journal of Health
    Geographics*, 4, 11.

    Examples
    --------
    >>> P = [(float(i), 0.0) for i in range(5)]
    >>> W = [[1 if abs(i - j) == 1 else 0 for j in range(5)] for i in range(5)]
    >>> r = flexscan(P, [8, 7, 1, 1, 1], [10] * 5, W, k=3, nsim=0)
    >>> r.clusters[0]["zone"], round(r.clusters[0]["llr"], 6)
    ([0, 1], 7.166736)
    """
    if kind not in ("poisson", "binomial"):
        raise ValueError("kind must be poisson or binomial")
    y, pp = _vec(cases), _vec(pop)
    ty, tpop = ssum(y), ssum(pp)
    e = [v * ty / tpop for v in pp] if ex is None else _vec(ex)
    zones = flex_zones(coords, W, k)
    ein = [ssum(e[j] for j in z) for z in zones]
    pin = [ssum(pp[j] for j in z) for z in zones]

    def stats(yy):
        out = []
        for z, ei, pi in zip(zones, ein, pin):
            yin = ssum(yy[j] for j in z)
            if kind == "poisson":
                out.append(_stat_poisson(yin, ty, ei, ty - ei, 1))
            else:
                out.append(_llr_binom(yin, ty, pi, tpop, 1))
        return out

    tobs = stats(y)
    tmax = (
        [max(stats([float(v) for v in _multinomial(ty, e, seed, s)])) for s in range(1, nsim + 1)] if nsim > 1 else []
    )
    pv = _pvals(tobs, tmax)
    sig = [i for i, p in enumerate(pv) if p <= alpha] or [min(range(len(pv)), key=lambda i: (pv[i], -tobs[i]))]
    sig.sort(key=lambda i: -tobs[i])
    keep, used = [], set()
    for i in sig:
        if not used & set(zones[i]):
            keep.append(i)
            used |= set(zones[i])
    clusters = [
        {"zone": zones[i], "llr": tobs[i], "pvalue": pv[i], "cases": ssum(y[j] for j in zones[i]), "expected": ein[i]}
        for i in keep
    ]
    return RichResult(payload={"clusters": clusters, "zones": zones, "tobs": tobs, "pvalues": pv})


def _perm(n, seed, stream):
    u = [float(v) for v in random_uniform(n, seed=seed, stream=stream)]
    p = list(range(n))
    for t in range(n - 1):
        k = t + int(u[t] * (n - t))
        p[t], p[k] = p[k], p[t]
    return p


def normal_scan(
    coords, values, *, pop=None, ubpop: float = 0.5, direction: str = "high", nsim: int = 499, seed: int = 1
) -> RichResult:
    r"""Normal-model spatial scan statistic for a continuous outcome (Kulldorff, Huang and Konty 2009).

    One observation per location. For circular zones (nearest-neighbour
    prefixes with population share at most ``ubpop``; ``pop`` defaults to 1
    per location) the maximum-likelihood variance under the alternative is
    ``s_z^2 = [sum_in (x - m_in)^2 + sum_out (x - m_out)^2] / N`` and ``LLR =
    N ln(s_0 / s_z)`` with ``s_0^2`` the overall variance, counted only when
    ``m_in > m_out`` (``direction="high"``), ``<`` (``"low"``) or either
    (``"both"``). The p-value of the maximum uses ``nsim`` random
    permutations of the values (Philox streams ``1..nsim``).

    References
    ----------
    Kulldorff, M., Huang, L. and Konty, K. (2009). A scan statistic for
    continuous data based on the normal probability model. *International
    Journal of Health Geographics*, 8, 58.

    Examples
    --------
    >>> P = [(float(i), 0.0) for i in range(6)]
    >>> r = normal_scan(P, [5.0, 6.0, 1.0, 2.0, 1.5, 0.5], nsim=0)
    >>> r.zone, round(r.llr, 6)
    ([0, 1], 8.07615)
    """
    P = _pts(coords)
    x = _vec(values)
    N = len(x)
    pp = [1.0] * N if pop is None else _vec(pop)
    nn = _nn(P, pp, ubpop)
    tot, tot2 = ssum(x), ssum(v * v for v in x)

    def best(xx):
        mu = tot / N
        s0 = tot2 / N - mu * mu
        bt, bz = 0.0, None
        for lst in nn:
            si = si2 = 0.0
            for m, j in enumerate(lst, 1):
                si += xx[j]
                si2 += xx[j] ** 2
                if m == N:
                    continue
                mi, mo = si / m, (tot - si) / (N - m)
                if (direction == "high" and mi <= mo) or (direction == "low" and mi >= mo) or mi == mo:
                    continue
                sz = (si2 - m * mi * mi + (tot2 - si2) - (N - m) * mo * mo) / N
                t = N * 0.5 * (math.log(s0) - math.log(sz)) if sz > 0 else math.inf
                if t > bt:
                    bt, bz = t, lst[:m]
        return bt, bz

    if direction not in ("high", "low", "both"):
        raise ValueError("direction must be high, low or both")
    t, z = best(x)
    tmax = [best([x[k] for k in _perm(N, seed, s)])[0] for s in range(1, nsim + 1)]
    return RichResult(
        payload={
            "zone": z,
            "llr": t,
            "pvalue": _pvals([t], tmax)[0],
            "mean_in": ssum(x[j] for j in z) / len(z) if z else math.nan,
        }
    )


def cuzick_edwards(coords, case, k: int = 1, *, nsim: int = 999, seed: int = 1) -> RichResult:
    r"""Cuzick-Edwards ``T_k``: the number of case-case pairs among each case's ``k`` nearest neighbours.

    ``T_k = sum_i sum_{j in kNN(i)} d_i d_j`` with ``d`` the case indicator
    (neighbours by Euclidean distance, ties by index). Under random labelling
    ``E(T_k) = k n_1 (n_1 - 1) / (n - 1)``; the one-sided p-value compares
    ``T_k`` with ``nsim`` random relabellings (Philox permutations).

    References
    ----------
    Cuzick, J. and Edwards, R. (1990). Spatial clustering for inhomogeneous
    populations. *Journal of the Royal Statistical Society B*, 52(1), 73-104.

    Examples
    --------
    >>> P = [(0, 0), (1, 0), (10, 0), (11, 0), (20, 0), (21, 0)]
    >>> r = cuzick_edwards(P, [1, 1, 0, 0, 1, 0], k=1, nsim=0)
    >>> r.statistic, r.expected
    (2, 1.2)
    """
    Pt = _pts(coords)
    d = [int(v) for v in case]
    n = len(d)
    nb = [sorted((j for j in range(n) if j != i), key=lambda j: (math.dist(Pt[i], Pt[j]), j))[:k] for i in range(n)]

    def T(lab):
        return sum(1 for i in range(n) if lab[i] for j in nb[i] if lab[j])

    t = T(d)
    n1 = sum(d)
    sims = []
    for s in range(1, nsim + 1):
        p = _perm(n, seed, s)
        sims.append(T([d[p[i]] for i in range(n)]))
    return RichResult(
        payload={
            "statistic": t,
            "expected": k * n1 * (n1 - 1) / (n - 1),
            "pvalue": (1 + sum(1 for v in sims if v >= t)) / (nsim + 1) if nsim else math.nan,
        }
    )


def lawson_waller(cases, expected, exposure, *, conditional: bool = True) -> RichResult:
    r"""Focused score test for raised incidence near a putative source (Waller et al. 1992; Lawson 1993).

    Under ``O_i ~ Poisson(E_i (1 + rho c_i))`` the score statistic for ``rho
    = 0`` is ``U = sum c_i (O_i - E_i)`` with ``c_i`` the exposure (e.g.
    inverse distance to the source). ``conditional=True`` rescales ``E`` to
    the observed total and uses the multinomial variance ``sum c^2 E - (sum c
    E)^2 / sum E``; otherwise the Poisson variance ``sum c^2 E``. Returns
    ``U``, its variance, ``z`` and the one-sided p-value.

    References
    ----------
    Waller, L. A., Turnbull, B. W., Clark, L. C. and Nasca, P. (1992).
    Chronic disease surveillance and testing of clustering of disease and
    exposure. *Environmetrics*, 3(3), 281-300.
    Waller, L. A. and Gotway, C. A. (2004). *Applied Spatial Statistics for
    Public Health Data*. Wiley, section 7.6.

    Examples
    --------
    >>> r = lawson_waller([4, 2, 1, 1], [2, 2, 2, 2], [1.0, 0.5, 0.25, 0.25])
    >>> r.U, r.variance
    (1.5, 0.75)
    """
    O_, E, c = _vec(cases), _vec(expected), _vec(exposure)
    if conditional:
        E = [v * ssum(O_) / ssum(E) for v in E]
    U = ssum(ci * (o - e) for ci, o, e in zip(c, O_, E))
    V = ssum(ci * ci * e for ci, e in zip(c, E))
    if conditional:
        V -= ssum(ci * e for ci, e in zip(c, E)) ** 2 / ssum(E)
    z = U / math.sqrt(V)
    return RichResult(payload={"U": U, "variance": V, "z": z, "pvalue": 1 - float(pnorm(z))})


def fixed_circle_scan(
    coords,
    cases,
    pop,
    radii,
    *,
    overlap: float = 0.2,
    method: str = "gam",
    alpha: float = 0.002,
    nsim: int = 99,
    seed: int = 1,
) -> RichResult:
    r"""Fixed-radius moving-circle scans: Openshaw's GAM and the Rushton-Lolonis Monte Carlo scan.

    For each radius ``r`` circles are centred on a square grid with spacing
    ``overlap * r`` covering the bounding box of ``coords``. Each circle's
    observed cases ``O`` and expected ``E = pop_in Y / pop_total`` are
    tested: ``gam`` uses the Poisson tail ``P(X >= O | E)`` (Openshaw et al.
    1987); ``rushton_lolonis`` the Monte Carlo p-value ``(1 + #{O_sim >=
    O}) / (nsim + 1)`` from ``nsim`` multinomial redistributions of the cases
    proportional to population (Rushton and Lolonis 1996). Circles with
    p-value at most ``alpha`` are returned.

    References
    ----------
    Openshaw, S., Charlton, M., Wymer, C. and Craft, A. (1987). A Mark 1
    Geographical Analysis Machine for the automated analysis of point data
    sets. *International Journal of Geographical Information Systems*, 1(4),
    335-358.
    Rushton, G. and Lolonis, P. (1996). Exploratory spatial analysis of birth
    defect rates in an urban population. *Statistics in Medicine*, 15(7-9),
    717-726.

    Examples
    --------
    >>> P = [(0.0, 0.0), (0.5, 0.0), (5.0, 5.0), (9.0, 1.0)]
    >>> r = fixed_circle_scan(P, [6, 5, 1, 0], [10, 10, 10, 10], [1.0], overlap=0.5, alpha=0.05)
    >>> sorted({tuple(c["members"]) for c in r.circles})
    [(0, 1)]
    """
    if method not in ("gam", "rushton_lolonis"):
        raise ValueError("method must be gam or rushton_lolonis")
    P = _pts(coords)
    y, pp = _vec(cases), _vec(pop)
    Y, tp = ssum(y), ssum(pp)
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    sims = [_multinomial(Y, pp, seed, s) for s in range(1, nsim + 1)] if method == "rushton_lolonis" else []
    out, ntest = [], 0
    for r in radii:
        step = overlap * r
        nx = int(math.floor((max(xs) - min(xs)) / step + 1e-9)) + 1
        ny = int(math.floor((max(ys) - min(ys)) / step + 1e-9)) + 1
        for a in range(nx):
            for b in range(ny):
                cx, cy = min(xs) + a * step, min(ys) + b * step
                mem = [j for j, p in enumerate(P) if math.hypot(p[0] - cx, p[1] - cy) <= r]
                if not mem:
                    continue
                ntest += 1
                O_ = ssum(y[j] for j in mem)
                E = ssum(pp[j] for j in mem) * Y / tp
                if method == "gam":
                    pv = 1.0 if O_ <= 0 else 1 - float(gammaincc(O_, E))
                else:
                    pv = (1 + sum(1 for s in sims if ssum(s[j] for j in mem) >= O_)) / (nsim + 1)
                if pv <= alpha:
                    out.append(
                        {
                            "center": (cx, cy),
                            "radius": float(r),
                            "members": mem,
                            "observed": O_,
                            "expected": E,
                            "pvalue": pv,
                        }
                    )
    return RichResult(payload={"circles": out, "n_circles": ntest})


def cheatsheet() -> str:
    return (
        "elliptic_scan / flexscan / normal_scan / cuzick_edwards / lawson_waller / fixed_circle_scan -> "
        "cluster detection beyond the circular scan."
    )
