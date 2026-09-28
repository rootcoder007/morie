# morie.fn -- function file (rootcoder007/morie)
"""Spatial cluster detection: Kulldorff circular scan, Besag-Newell, Tango's index and Stone's test."""

from __future__ import annotations

import bisect
import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import pchisq
from ._sci_core import gammaincc

__all__ = ["scan_zones", "kulldorff_scan", "besag_newell", "tango_test", "stone_test"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).tolist()]


def _pts(coords):
    return [tuple(float(v) for v in r) for r in np.asarray(coords, dtype=float).tolist()]


def _nn(P, pop, ubpop):
    """Per-centroid nearest-neighbour lists whose cumulative population stays within ubpop x total (smerc)."""
    n = len(P)
    tp = ssum(pop)
    out = []
    for i in range(n):
        order = sorted(range(n), key=lambda j: (math.dist(P[i], P[j]), j))
        cs, keep = 0.0, []
        for j in order:
            cs += pop[j]
            if cs <= tp * ubpop:
                keep.append(j)
            else:
                break
        out.append(keep)
    return out


def scan_zones(coords, pop, *, ubpop: float = 0.5):
    r"""Distinct circular scan zones (Kulldorff 1997), as ``smerc::scan.zones``.

    For each centroid the nearest regions are added one at a time while the
    cumulative population is at most ``ubpop`` times the total; every prefix
    is a candidate zone, and zones with the same region set are kept once
    (first occurrence).  Regions are 0-based.

    Examples
    --------
    >>> scan_zones([(0, 0), (1, 0), (5, 0)], [1, 1, 1], ubpop=0.7)
    [[0], [0, 1], [1], [2], [2, 1]]
    """
    nn = _nn(_pts(coords), _vec(pop), ubpop)
    seen, out = set(), []
    for lst in nn:
        for k in range(1, len(lst) + 1):
            z = lst[:k]
            key = frozenset(z)
            if key not in seen:
                seen.add(key)
                out.append(z)
    return out


def _llr_poisson(yin, ty, ein, min_cases):
    if yin < min_cases:
        return 0.0
    eout = ty - ein
    yout = ty - yin
    if yin / ein <= (yout / eout if eout > 0 else float("inf")):
        return 0.0
    t = yin * (math.log(yin) - math.log(ein))
    if yout > 0:
        t += yout * (math.log(yout) - math.log(eout))
    return t


def _llr_binom(yin, ty, popin, tpop, min_cases):
    if yin < min_cases:
        return 0.0
    popout = tpop - popin
    yout = ty - yin
    if popout <= 0 or yin / popin <= yout / popout:
        return 0.0

    def xlogy(a, b):
        return a * math.log(b) if a > 0 else 0.0

    t = (
        xlogy(yin, yin)
        - xlogy(yin, popin)
        + xlogy(popin - yin, popin - yin)
        - xlogy(popin - yin, popin)
        + xlogy(yout, yout)
        - xlogy(yout, popout)
        + xlogy(popout - yout, popout - yout)
        - xlogy(popout - yout, popout)
        - xlogy(ty, ty)
        - xlogy(tpop - ty, tpop - ty)
        + xlogy(tpop, tpop)
    )
    return t if t == t else 0.0


def _zone_stats(nn, cases, ex, pop, kind, min_cases):
    ty = ssum(cases)
    mult = ty / ssum(ex)
    tp = ssum(pop)
    stats = []
    for lst in nn:
        yin = ein = pin = 0.0
        row = []
        for j in lst:
            yin += cases[j]
            ein += ex[j]
            pin += pop[j]
            if kind == "poisson":
                row.append(_llr_poisson(yin, ty, ein * mult, min_cases))
            else:
                row.append(_llr_binom(yin, ty, pin, tp, min_cases))
        stats.append(row)
    return stats


def _noc(nn, stats):
    """Non-overlapping most likely and secondary clusters (smerc ``noc_nn``)."""
    stats = [list(r) for r in stats]
    remaining = [i for i in range(len(nn)) if nn[i]]
    clusts, tobs = [], []
    cur = 1.0
    while remaining and cur > 0:
        best_i, best_t, best_k = None, -1.0, 0
        for i in remaining:
            if not stats[i]:
                continue
            k = max(range(len(stats[i])), key=lambda t: (stats[i][t], -t))
            if stats[i][k] > best_t:
                best_i, best_t, best_k = i, stats[i][k], k
        if best_i is None:
            break
        clusts.append(nn[best_i][: best_k + 1])
        cur = best_t
        tobs.append(best_t)
        used = {j for c in clusts for j in c}
        remaining = [i for i in remaining if i not in used]
        for i in remaining:
            for t, j in enumerate(nn[i]):
                if j in used:
                    stats[i] = stats[i][:t]
                    break
    return clusts, tobs


def _multinomial(total, prob, seed, stream):
    cum, s = [], 0.0
    tp = ssum(prob)
    for p in prob:
        s += p / tp
        cum.append(s)
    cum[-1] = 1.0
    out = [0] * len(prob)
    for u in random_uniform(int(total), seed=seed, stream=stream):
        out[bisect.bisect_right(cum, float(u))] += 1
    return out


def kulldorff_scan(
    coords,
    cases,
    pop,
    *,
    ex=None,
    kind: str = "poisson",
    ubpop: float = 0.5,
    nsim: int = 499,
    alpha: float = 0.1,
    min_cases: int = 2,
    seed: int = 1,
) -> RichResult:
    r"""Kulldorff's circular spatial scan statistic (Kulldorff 1997), as ``smerc::scan.test``.

    Candidate zones are the nearest-neighbour prefixes of :func:`scan_zones`.
    ``poisson``: ``LLR = y log(y/E) + (Y - y) log((Y - y)/(Y - E))`` for
    zones with more than expected cases (``E`` scaled to the total ``Y``);
    ``binomial``: the Bernoulli likelihood ratio in the zone populations; 0
    below ``min_cases``.  The most likely cluster and non-overlapping
    secondary clusters are chosen greedily; each p-value is ``(1 + #{T_sim >=
    T}) / (1 + nsim)`` over the maxima of ``nsim`` multinomial redistributions
    of the ``Y`` cases proportional to ``ex`` (Philox streams ``1..nsim`` of
    ``seed``).  ``clusters`` keep those with p-value at most ``alpha`` (the
    most likely one if none).

    References
    ----------
    Kulldorff, M. (1997). A spatial scan statistic. *Communications in
    Statistics - Theory and Methods*, 26(6), 1481-1496.

    Examples
    --------
    >>> P = [(float(i), 0.0) for i in range(6)]
    >>> r = kulldorff_scan(P, [9, 8, 1, 1, 1, 0], [10] * 6, nsim=0)
    >>> r.all_zones[0], round(r.all_tobs[0], 6)
    ([0, 1], 11.438622)
    """
    if kind not in ("poisson", "binomial"):
        raise ValueError("kind must be poisson or binomial")
    P = _pts(coords)
    y, pp = _vec(cases), _vec(pop)
    e = [v * ssum(y) / ssum(pp) for v in pp] if ex is None else _vec(ex)
    nn = _nn(P, pp, ubpop)
    zones, tobs = _noc(nn, _zone_stats(nn, y, e, pp, kind, min_cases))
    if nsim > 0:
        tmax = []
        for s in range(1, nsim + 1):
            ys = [float(v) for v in _multinomial(ssum(y), e, seed, s)]
            st = _zone_stats(nn, ys, e, pp, kind, min_cases)
            tmax.append(max((v for r in st for v in r), default=0.0))
        pv = [(1 + sum(1 for t in tmax if t >= x)) / (nsim + 1) for x in tobs]
    else:
        pv = [1.0] * len(tobs)
    sig = [i for i, p in enumerate(pv) if p <= alpha] or [min(range(len(pv)), key=lambda i: pv[i])]
    clusters = [
        {
            "zone": zones[i],
            "llr": tobs[i],
            "pvalue": pv[i],
            "cases": ssum(y[j] for j in zones[i]),
            "expected": ssum(e[j] for j in zones[i]) * ssum(y) / ssum(e),
        }
        for i in sig
    ]
    return RichResult(payload={"clusters": clusters, "all_zones": zones, "all_tobs": tobs, "all_pvalues": pv})


def besag_newell(coords, cases, pop, k: int, *, expected=None) -> RichResult:
    r"""Besag and Newell (1991) cluster test, as ``SpatialEpi::besag_newell``.

    For each region the nearest regions (itself first) are pooled until they
    hold at least ``k`` cases; with ``m`` regions pooled, ``k_obs`` cases
    reached and ``E`` their expected count, the p-value is ``P(Poisson(E) >=
    k_obs)`` as SpatialEpi computes it (1 when fewer than ``k`` cases exist
    in total).  Returns ``m_values``, ``k_values`` (cases
    reached) and ``p_values``.

    References
    ----------
    Besag, J. and Newell, J. (1991). The detection of clusters in rare
    diseases. *Journal of the Royal Statistical Society A*, 154(1), 143-155.

    Examples
    --------
    >>> r = besag_newell([(0, 0), (1, 0), (2, 0), (9, 0)], [3, 2, 0, 1], [10, 10, 10, 10], 4)
    >>> r.m_values
    [2, 2, 3, 4]
    """
    P = _pts(coords)
    y, pp = _vec(cases), _vec(pop)
    n = len(y)
    E = [v * ssum(y) / ssum(pp) for v in pp] if expected is None else _vec(expected)
    mv, kv, pv = [], [], []
    for i in range(n):
        order = sorted(range(n), key=lambda j: (math.dist(P[i], P[j]), j))
        c = e = 0.0
        m = 0
        for j in order:
            m += 1
            c += y[j]
            e += E[j]
            if c >= k:
                break
        mv.append(m)
        kv.append(c)
        pv.append(float(1.0 - gammaincc(c, e)) if c >= k else 1.0)
    return RichResult(payload={"m_values": mv, "k_values": kv, "p_values": pv})


def tango_test(cases, pop, W) -> RichResult:
    r"""Tango's (1995) index of spatial clustering, as ``smerc::tango.test``.

    With case and population proportions ``r`` and ``p``, ``C = (r - p)' W (r
    - p)`` split into ``gof = sum (r - p)^2`` and ``sa = (r - p)'(W - I)(r -
    p)``, and Tango's chi-square approximation: with ``V = diag(p) - p p'``,
    ``E(C) = tr(WV)/Y``, ``Var(C) = 2 tr((WV)^2)/Y^2``, skewness ``2 sqrt 2
    tr((WV)^3) / tr((WV)^2)^{3/2}``, ``df = 8 / skew^2``, statistic ``df +
    z sqrt(2 df)`` and its upper chi-square p-value.

    References
    ----------
    Tango, T. (1995). A class of tests for detecting 'general' and 'focused'
    clustering of rare diseases. *Statistics in Medicine*, 14(21-22),
    2323-2334.

    Examples
    --------
    >>> W = [[1, 0.5, 0.1], [0.5, 1, 0.5], [0.1, 0.5, 1]]
    >>> round(tango_test([5, 1, 0], [10, 10, 10], W).tstat, 6)
    0.327778
    """
    y, pp = _vec(cases), _vec(pop)
    Wm = [[float(v) for v in r] for r in W]
    n = len(y)
    Y = ssum(y)
    r = [v / Y for v in y]
    p = [v / ssum(pp) for v in pp]
    ee = [a - b for a, b in zip(r, p)]
    gof = ssum(v * v for v in ee)
    sa = ssum(ee[i] * (Wm[i][j] - (1.0 if i == j else 0.0)) * ee[j] for i in range(n) for j in range(n))
    V = [[(p[i] if i == j else 0.0) - p[i] * p[j] for j in range(n)] for i in range(n)]
    WV = [[ssum(Wm[i][t] * V[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    WV2 = [[ssum(WV[i][t] * WV[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    tr1 = ssum(WV[i][i] for i in range(n))
    tr2 = ssum(WV2[i][i] for i in range(n))
    tr3 = ssum(WV2[i][t] * WV[t][i] for i in range(n) for t in range(n))
    ec, vc = tr1 / Y, 2.0 * tr2 / Y**2
    skc = 2.0 * math.sqrt(2.0) * tr3 / tr2**1.5
    dfc = 8.0 / skc**2
    t = gof + sa
    tchi = dfc + (t - ec) / math.sqrt(vc) * math.sqrt(2.0 * dfc)
    return RichResult(
        payload={
            "tstat": t,
            "gof": gof,
            "sa": sa,
            "tstat_chisq": tchi,
            "dfc": dfc,
            "pvalue_chisq": 1.0 - pchisq(tchi, dfc),
        }
    )


def stone_test(cases, expected, order, *, nsim: int = 0, seed: int = 1) -> RichResult:
    r"""Stone's (1988) test of raised risk near a putative source.

    With regions sorted by ``order`` (increasing distance from the source),
    ``T = max_k sum_{j <= k} O_j / sum_{j <= k} E_j``; the attaining ``k`` is
    returned, and with ``nsim > 0`` a Monte Carlo p-value from multinomial
    redistributions of the total count proportional to ``expected`` (Philox
    streams ``1..nsim``).

    References
    ----------
    Stone, R. A. (1988). Investigations of excess environmental risks around
    putative sources: statistical problems and a proposed test. *Statistics
    in Medicine*, 7(6), 649-660.

    Examples
    --------
    >>> r = stone_test([4, 2, 1, 1], [1.0, 2.0, 2.0, 3.0], [0, 1, 2, 3])
    >>> r.statistic, r.k
    (4.0, 1)
    """
    y, E = _vec(cases), _vec(expected)
    idx = list(order)

    def stat(v):
        co = ce = 0.0
        best, bk = -1.0, 0
        for k, j in enumerate(idx, start=1):
            co += v[j]
            ce += E[j]
            if co / ce > best:
                best, bk = co / ce, k
        return best, bk

    T, k = stat(y)
    out = {"statistic": T, "k": k}
    if nsim > 0:
        sims = [stat([float(v) for v in _multinomial(ssum(y), E, seed, s)])[0] for s in range(1, nsim + 1)]
        out["pvalue"] = (1 + sum(1 for s in sims if s >= T)) / (nsim + 1)
    return RichResult(payload=out)


def cheatsheet() -> str:
    return "kulldorff_scan / besag_newell / tango_test / stone_test -> spatial cluster detection."
