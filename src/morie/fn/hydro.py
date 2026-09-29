# morie.fn -- function file (rootcoder007/morie)
"""Hydrology: flood and low-flow frequency (L-moments, log-Pearson III), flow duration and IDF
curves, unit hydrographs and runoff convolution, baseflow separation and hydrograph summaries,
Horton stream-network analysis, channel slope and meander metrics, Darcy flow, depression breaching
and height above nearest drainage."""

from __future__ import annotations

import heapq
import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rrng_core import pgamma, qgamma, qnorm
from ._sci_core import gammaln
from .demops import _D8, _OFF, _grid, _ok, _receivers

__all__ = [
    "sample_lmoments",
    "flood_frequency",
    "low_flow_frequency",
    "flow_duration_curve",
    "idf_fit",
    "unit_hydrograph",
    "convolve_runoff",
    "baseflow_filter",
    "hydrograph_summary",
    "stream_segments",
    "channel_slope",
    "meander_metrics",
    "darcy_flow",
    "breach_depressions",
    "height_above_drainage",
]

_EU = 0.57721566490153286
_DL2, _DL3 = math.log(2.0), math.log(3.0)


def sample_lmoments(x) -> list:
    r"""Sample L-moments ``[l1, l2, t3, t4]`` from unbiased probability-weighted moments (Hosking 1990), as ``lmom::samlmu``.

    ``b_r = n^-1 sum_i x_(i) prod_{k=1..r} (i - k) / (n - k)``; ``l1 = b0``,
    ``l2 = 2 b1 - b0``, ``l3 = 6 b2 - 6 b1 + b0``, ``l4 = 20 b3 - 30 b2 + 12 b1
    - b0``, ``t_r = l_r / l2``.

    References
    ----------
    Hosking, J. R. M. (1990). L-moments: analysis and estimation of
    distributions using linear combinations of order statistics. *JRSS B*,
    52(1), 105-124.

    Examples
    --------
    >>> [round(v, 12) for v in sample_lmoments([1, 2, 3, 4, 10])]
    [4.0, 2.0, 0.5, 0.5]
    """
    s = sorted(float(v) for v in x)
    n = len(s)
    b = [ssum(s) / n]
    for r in range(1, 4):
        b.append(ssum(s[i] * math.prod((i - k) / (n - 1 - k) for k in range(r)) for i in range(n)) / n)
    l2 = 2 * b[1] - b[0]
    return [b[0], l2, (6 * b[2] - 6 * b[1] + b[0]) / l2, (20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]) / l2]


def _pelgev(l1, l2, t3):
    """Hosking's PELGEV (lmom::pelgev): rational approximations, Newton below tau3 = -0.8."""
    if t3 > 0:
        z = 1 - t3
        g = (-1 + z * (1.59921491 + z * (-0.48832213 + z * 0.01573152))) / (1 + z * (-0.64363929 + z * 0.08985247))
        if abs(g) < 1e-5:
            a = l2 / _DL2
            return [l1 - _EU * a, a, 0.0]
    else:
        g = (0.28377530 + t3 * (-1.21096399 + t3 * (-2.50728214 + t3 * (-1.13455566 + t3 * -0.07138022)))) / (
            1 + t3 * (2.06189696 + t3 * (1.31912239 + t3 * 0.25077104))
        )
        if t3 < -0.8:
            if t3 <= -0.97:
                g = 1 - math.log(1 + t3) / _DL2
            t0 = (t3 + 3) * 0.5
            for _ in range(20):
                x2, x3 = 2.0 ** (-g), 3.0 ** (-g)
                xx2, xx3 = 1 - x2, 1 - x3
                deriv = (xx2 * x3 * _DL3 - xx3 * x2 * _DL2) / (xx2 * xx2)
                gold = g
                g = g - (xx3 / xx2 - t0) / deriv
                if abs(g - gold) <= 1e-6 * g:
                    break
    gam = math.exp(gammaln(1 + g))
    a = l2 * g / (gam * (1 - 2.0 ** (-g)))
    return [l1 - a * (1 - gam) / g, a, g]


def _quagev(f, p):
    return p[0] - p[1] * math.log(-math.log(f)) if p[2] == 0 else p[0] + p[1] / p[2] * (1 - (-math.log(f)) ** p[2])


def _quape3(f, mu, sd, g):
    if abs(g) <= 1e-8:
        return mu + sd * float(qnorm(f))
    al, be = 4 / g**2, abs(0.5 * sd * g)
    if g > 0:
        return mu - al * be + be * max(0.0, float(qgamma(f, al)))
    return mu + al * be - be * max(0.0, float(qgamma(1 - f, al)))


def _moments(v):
    n = len(v)
    m = ssum(v) / n
    s = math.sqrt(ssum((a - m) ** 2 for a in v) / (n - 1))
    g = n * ssum((a - m) ** 3 for a in v) / ((n - 1) * (n - 2) * s**3)
    return m, s, g


def flood_frequency(amax, *, dist: str = "gev", return_periods=(2, 5, 10, 25, 50, 100)) -> RichResult:
    r"""Flood frequency analysis of annual maxima: T-year floods ``Q_T = F^-1(1 - 1/T)``.

    ``gev`` and ``gumbel`` are fitted by L-moments (Hosking and Wallis 1997;
    the GEV shape by Hosking's rational approximations, as ``lmom::pelgev``,
    in ``lmom``'s parameterisation ``xi + alpha (1 - (-log F)^k) / k``);
    ``lp3`` is Bulletin 17B's log-Pearson type III: mean, standard deviation
    and bias-corrected skew ``G = n sum (y - ybar)^3 / ((n-1)(n-2) s^3)`` of
    ``y = log10 Q``, quantiles by the Pearson III quantile (gamma).

    References
    ----------
    Hosking, J. R. M. and Wallis, J. R. (1997). *Regional Frequency
    Analysis: An Approach Based on L-Moments*. Cambridge University Press.
    Interagency Advisory Committee on Water Data (1982). *Guidelines for
    Determining Flood Flow Frequency*, Bulletin 17B. USGS.

    Examples
    --------
    >>> r = flood_frequency([120, 95, 310, 180, 150, 220, 90, 260, 140, 175], dist="gumbel")
    >>> round(r.quantiles[0], 6)
    161.232848
    """
    q = [float(v) for v in amax]
    T = [float(t) for t in return_periods]
    F = [1 - 1 / t for t in T]
    if dist in ("gev", "gumbel"):
        l1, l2, t3, _ = sample_lmoments(q)
        if dist == "gumbel":
            a = l2 / _DL2
            par = [l1 - _EU * a, a]
            qs = [par[0] - par[1] * math.log(-math.log(f)) for f in F]
        else:
            par = _pelgev(l1, l2, t3)
            qs = [_quagev(f, par) for f in F]
    elif dist == "lp3":
        par = list(_moments([math.log10(v) for v in q]))
        qs = [10 ** _quape3(f, *par) for f in F]
    else:
        raise ValueError("dist must be gev, gumbel or lp3")
    return RichResult(payload={"params": par, "return_periods": T, "quantiles": qs, "dist": dist})


def low_flow_frequency(flows, years, *, d: int = 7, T: float = 10.0, dist: str = "weibull") -> RichResult:
    r"""Low-flow frequency: the ``d``-day, ``T``-year low flow (e.g. 7Q10) from daily flows.

    The annual series is the minimum over each year of the ``d``-day moving
    average (windows within the year). ``weibull`` fits the three-parameter
    Weibull by L-moments (as ``lmom::pelwei``, the reversed GEV) and returns
    its quantile at non-exceedance ``1/T``; ``lp3`` fits log-Pearson III as in
    :func:`flood_frequency`.

    References
    ----------
    Riggs, H. C. (1972). Low-flow investigations. *Techniques of
    Water-Resources Investigations*, Book 4, Chapter B1. USGS.

    Examples
    --------
    >>> q = [5 + (i % 30) + 3 * (i // 365) for i in range(365 * 6)]
    >>> r = low_flow_frequency(q, [i // 365 for i in range(365 * 6)], d=7, T=2, dist="lp3")
    >>> [round(v, 6) for v in r.annual_minima[:3]]
    [8.0, 11.0, 14.0]
    """
    q = [float(v) for v in flows]
    yrs = list(years)
    mins = []
    for yv in sorted(set(yrs), key=yrs.index):
        s = [q[i] for i in range(len(q)) if yrs[i] == yv]
        if len(s) >= d:
            mins.append(min(ssum(s[i : i + d]) / d for i in range(len(s) - d + 1)))
    f = 1 / T
    if dist == "weibull":
        l1, l2, t3, _ = sample_lmoments(mins)
        pg = _pelgev(-l1, l2, -t3)
        delta, beta = 1 / pg[2], pg[1] / pg[2]
        par = [-pg[0] - beta, beta, delta]
        val = par[0] + par[1] * (-math.log(1 - f)) ** (1 / par[2])
    elif dist == "lp3":
        par = list(_moments([math.log10(v) for v in mins]))
        val = 10 ** _quape3(f, *par)
    else:
        raise ValueError("dist must be weibull or lp3")
    return RichResult(payload={"annual_minima": mins, "params": par, "quantile": val, "d": d, "T": T})


def flow_duration_curve(flows, probs=None) -> RichResult:
    r"""Flow duration curve: flows sorted descending against exceedance probability ``p_i = i / (n + 1)`` (Weibull plotting position).

    ``probs`` (in (0, 1)) are read off by linear interpolation in ``p``
    (constant beyond the end points), e.g. ``Q95`` at ``0.95``.

    References
    ----------
    Vogel, R. M. and Fennessey, N. M. (1994). Flow-duration curves. I: New
    interpretation and confidence intervals. *Journal of Water Resources
    Planning and Management*, 120(4), 485-504.

    Examples
    --------
    >>> flow_duration_curve([3, 1, 2, 4], [0.5]).quantiles
    [2.5]
    """
    s = sorted((float(v) for v in flows), reverse=True)
    n = len(s)
    p = [(i + 1) / (n + 1) for i in range(n)]
    qs = []
    for pr in probs or []:
        if pr <= p[0]:
            qs.append(s[0])
        elif pr >= p[-1]:
            qs.append(s[-1])
        else:
            k = int(pr * (n + 1)) - 1
            while p[k + 1] < pr:
                k += 1
            w = (pr - p[k]) / (p[k + 1] - p[k])
            qs.append(s[k] + w * (s[k + 1] - s[k]))
    return RichResult(payload={"flows": s, "exceedance": p, "quantiles": qs})


def idf_fit(durations, intensities, *, start=None, tol: float = 1e-13, maxit: int = 500) -> RichResult:
    r"""Fit the Sherman intensity-duration-frequency curve ``i = a / (t + b)^c`` by least squares (Levenberg-Marquardt).

    Starts from the log-linear fit with ``b = 0`` unless ``start = (a, b,
    c)``; stops when every accepted step changes the parameters by less than
    ``tol`` relatively, or no step decreases the residual sum of squares.

    References
    ----------
    Sherman, C. W. (1931). Frequency and intensity of excessive rainfalls at
    Boston, Massachusetts. *Transactions of the ASCE*, 95, 951-960.

    Examples
    --------
    >>> t = [5, 10, 15, 30, 60, 120]
    >>> r = idf_fit(t, [1000 / (v + 8) ** 0.7 for v in t])
    >>> round(r.a, 6), round(r.b, 6), round(r.c, 6)
    (1000.0, 8.0, 0.7)
    """
    t = [float(v) for v in durations]
    y = [float(v) for v in intensities]
    n = len(t)
    if start is None:
        lx = [math.log(v) for v in t]
        ly = [math.log(v) for v in y]
        mx, my = ssum(lx) / n, ssum(ly) / n
        c = -ssum((a - mx) * (b - my) for a, b in zip(lx, ly)) / ssum((a - mx) ** 2 for a in lx)
        par = [math.exp(my + c * mx), 0.0, c]
    else:
        par = [float(v) for v in start]

    def rss(p):
        if min(v + p[1] for v in t) <= 0:
            return math.inf
        return ssum((yi - p[0] / (ti + p[1]) ** p[2]) ** 2 for ti, yi in zip(t, y))

    lam, cur = 1e-3, rss(par)
    for _ in range(maxit):
        J, r = [], []
        for ti, yi in zip(t, y):
            u = ti + par[1]
            f = par[0] / u ** par[2]
            J.append([f / par[0], -par[2] * f / u, -f * math.log(u)])
            r.append(yi - f)
        JtJ = [[ssum(J[k][a] * J[k][b] for k in range(n)) for b in range(3)] for a in range(3)]
        Jtr = [ssum(J[k][a] * r[k] for k in range(n)) for a in range(3)]
        improved = False
        for _ in range(60):
            M = [[JtJ[a][b] * (1 + lam if a == b else 1) for b in range(3)] for a in range(3)]
            Mi = inverse(M)
            step = [ssum(float(Mi[a][b]) * Jtr[b] for b in range(3)) for a in range(3)]
            cand = [p + s for p, s in zip(par, step)]
            new = rss(cand)
            if new <= cur:
                improved = True
                break
            lam *= 10
        if not improved:
            break
        done = all(abs(st) <= tol * abs(p) for st, p in zip(step, cand))
        par, cur, lam = cand, new, max(lam / 10, 1e-12)
        if done:
            break
    return RichResult(
        payload={
            "a": par[0],
            "b": par[1],
            "c": par[2],
            "rss": cur,
            "fitted": [par[0] / (ti + par[1]) ** par[2] for ti in t],
        }
    )


def unit_hydrograph(
    *,
    method: str = "scs",
    area: float = 1.0,
    dt: float = 0.5,
    D: float = 1.0,
    tc: float | None = None,
    n: float | None = None,
    k: float | None = None,
) -> RichResult:
    r"""D-hour unit hydrograph ordinates (m^3/s per cm of excess rain over ``area`` km^2) at time step ``dt`` hours.

    ``scs``: SCS triangular hydrograph with lag ``0.6 tc``, time to peak ``Tp
    = D/2 + 0.6 tc``, peak ``qp = 2.08 A / Tp`` and base ``2.67 Tp`` (NRCS
    National Engineering Handbook, Part 630, Ch. 16). ``nash``: Nash cascade
    of ``n`` linear reservoirs with storage constant ``k`` hours; the D-hour UH
    is the S-curve difference ``[G(t/k; n) - G((t - D)/k; n)] / D`` with the
    gamma CDF ``G``, scaled by ``A / 0.36`` (1 cm over 1 km^2 in 1 h is
    ``1e4/3600`` m^3/s). Both have volume ``A`` x 1 cm.

    References
    ----------
    Nash, J. E. (1957). The form of the instantaneous unit hydrograph.
    *IAHS Publication*, 45(3), 114-121.
    USDA-NRCS (2007). *National Engineering Handbook*, Part 630, Chapter 16.

    Examples
    --------
    >>> r = unit_hydrograph(method="scs", area=10, dt=1, D=2, tc=5 / 0.6)
    >>> r.time_to_peak, round(r.peak, 6)
    (6.0, 3.466667)
    """
    if method == "scs":
        if tc is None:
            raise ValueError("scs needs tc")
        tp = D / 2 + 0.6 * tc
        qp = 2.08 * area / tp
        tb = 2.67 * tp
        m = int(math.ceil(tb / dt))
        times = [i * dt for i in range(m + 1)]
        q = [qp * t / tp if t <= tp else (qp * (tb - t) / (tb - tp) if t < tb else 0.0) for t in times]
        return RichResult(payload={"time": times, "ordinates": q, "time_to_peak": tp, "peak": qp, "base": tb})
    if method == "nash":
        if n is None or k is None:
            raise ValueError("nash needs n and k")
        s = area / 0.36

        def G(t):
            return float(pgamma(t / k, n)) if t > 0 else 0.0

        times, q = [0.0], [0.0]
        i = 1
        while True:
            t = i * dt
            v = s * (G(t) - G(t - D)) / D
            times.append(t)
            q.append(v)
            if t > D and G(t - D) > 1 - 1e-10:
                break
            i += 1
        j = max(range(len(q)), key=lambda z: q[z])
        return RichResult(
            payload={"time": times, "ordinates": q, "time_to_peak": times[j], "peak": q[j], "base": times[-1]}
        )
    raise ValueError("method must be scs or nash")


def convolve_runoff(excess, uh) -> list:
    r"""Direct runoff ``Q_n = sum_m P_m U_{n-m+1}``: discrete convolution of excess-rain depths (cm per D-hour block) with UH ordinates.

    Examples
    --------
    >>> convolve_runoff([1, 2], [0, 1, 3, 1, 0])
    [0.0, 1.0, 5.0, 7.0, 2.0, 0.0]
    """
    P = [float(v) for v in excess]
    U = [float(v) for v in uh]
    return [ssum(P[m] * U[j - m] for m in range(len(P)) if 0 <= j - m < len(U)) for j in range(len(P) + len(U) - 1)]


def baseflow_filter(flows, *, alpha: float = 0.925, passes: int = 3) -> RichResult:
    r"""Lyne-Hollick recursive digital filter baseflow separation (Nathan and McMahon 1990).

    Quickflow ``f_t = alpha f_{t-1} + (1 + alpha)/2 (q_t - q_{t-1})``, clamped
    to ``[0, q_t]``, baseflow ``q - f``; passes alternate forward, backward,
    forward, each on the previous pass's baseflow and starting with zero
    quickflow. Returns the baseflow and the baseflow index ``sum b / sum q``.

    References
    ----------
    Nathan, R. J. and McMahon, T. A. (1990). Evaluation of automated
    techniques for base flow and recession analyses. *Water Resources
    Research*, 26(7), 1465-1473.

    Examples
    --------
    >>> round(baseflow_filter([5, 5, 5, 5]).bfi, 12)
    1.0
    """
    q = [float(v) for v in flows]
    b = q[:]
    for ps in range(passes):
        s = b if ps % 2 == 0 else b[::-1]
        out = [s[0]]
        f = 0.0
        for t in range(1, len(s)):
            f = alpha * f + (1 + alpha) / 2 * (s[t] - s[t - 1])
            f = min(max(f, 0.0), s[t])
            out.append(s[t] - f)
        b = out if ps % 2 == 0 else out[::-1]
    return RichResult(payload={"baseflow": b, "quickflow": [a - c for a, c in zip(q, b)], "bfi": ssum(b) / ssum(q)})


def hydrograph_summary(flows, *, dt: float = 1.0, alpha: float = 0.925, min_recession: int = 3) -> RichResult:
    r"""Hydrograph summary: peak, time to peak, volume (trapezoidal, flow x ``dt``), baseflow index and recession constant.

    The recession constant ``k`` (``Q_{t+1} = k Q_t``) is ``exp`` of the
    least-squares slope of ``log Q`` on time pooled over every run of at
    least ``min_recession`` consecutive decreasing flows (each run centred).

    References
    ----------
    Tallaksen, L. M. (1995). A review of baseflow recession analysis.
    *Journal of Hydrology*, 165(1-4), 349-370.

    Examples
    --------
    >>> r = hydrograph_summary([1, 4, 8, 4, 2, 1, 0.5])
    >>> r.peak, r.time_to_peak, round(r.recession_constant, 12)
    (8.0, 2.0, 0.5)
    """
    q = [float(v) for v in flows]
    j = max(range(len(q)), key=lambda i: q[i])
    runs, cur = [], [0]
    for t in range(1, len(q)):
        if q[t] < q[t - 1] and q[t] > 0:
            cur.append(t)
        else:
            if len(cur) > min_recession:
                runs.append(cur)
            cur = [t]
    if len(cur) > min_recession:
        runs.append(cur)
    sxy = sxx = 0.0
    for r in runs:
        tm = ssum(r) / len(r)
        lm = ssum(math.log(q[t]) for t in r) / len(r)
        sxy += ssum((t - tm) * (math.log(q[t]) - lm) for t in r)
        sxx += ssum((t - tm) ** 2 for t in r)
    return RichResult(
        payload={
            "peak": q[j],
            "time_to_peak": j * dt,
            "volume": dt * (ssum(q) - 0.5 * (q[0] + q[-1])),
            "bfi": baseflow_filter(q, alpha=alpha).bfi,
            "recession_constant": math.exp(sxy / sxx) if sxx > 0 else math.nan,
            "n_recessions": len(runs),
        }
    )


def stream_segments(flowdir, acc, threshold: float, *, res: float = 1.0) -> RichResult:
    r"""Channel network extraction (accumulation ``>= threshold``), Strahler segments and Horton's laws.

    A segment of order ``w`` runs from a cell of order ``w`` without a donor
    of order ``w`` downstream until its receiver has another order (or it
    leaves the network); its length sums the D8 steps (``res`` or ``res
    sqrt 2``) including the step into the receiver. Horton's bifurcation and
    length ratios ``R_b``, ``R_L`` are ``10^-slope`` and ``10^slope`` of the
    least-squares lines of ``log10 N_w`` and ``log10 mean L_w`` on ``w``.

    References
    ----------
    Horton, R. E. (1945). Erosional development of streams and their drainage
    basins. *GSA Bulletin*, 56(3), 275-370.
    Strahler, A. N. (1957). Quantitative analysis of watershed geomorphology.
    *Transactions, AGU*, 38(6), 913-920.

    Examples
    --------
    >>> fd = [[2, 4, 8], [1, 4, 16], [1, 4, 16]]
    >>> from morie.fn.demops import flow_accumulation
    >>> r = stream_segments(fd, flow_accumulation(fd), 1)
    >>> r.counts, r.mean_lengths
    ([7, 1], [1.1183467321065985, 1.0])
    """
    from .demops import stream_order

    nr, nc = len(flowdir), len(flowdir[0])
    order = stream_order(flowdir, acc, threshold)
    rec = _receivers(flowdir)
    donors = {}
    for s, d in rec.items():
        if order[s[0]][s[1]] and order[d[0]][d[1]]:
            donors.setdefault(d, []).append(s)
    segs = []
    for i in range(nr):
        for j in range(nc):
            w = order[i][j]
            if not w or any(order[a][b] == w for a, b in donors.get((i, j), [])):
                continue
            cells, L, c = [(i, j)], 0.0, (i, j)
            while True:
                d = rec.get(c)
                if d is None or not order[d[0]][d[1]]:
                    break
                a, b = d[0] - c[0], d[1] - c[1]
                L += res * (math.sqrt(2.0) if a and b else 1.0)
                if order[d[0]][d[1]] != w:
                    break
                cells.append(d)
                c = d
            segs.append({"order": w, "cells": cells, "length": L})
    W = max((s["order"] for s in segs), default=0)
    counts = [sum(1 for s in segs if s["order"] == w) for w in range(1, W + 1)]
    mlen = [
        ssum(s["length"] for s in segs if s["order"] == w) / c if c else 0.0 for w, c in zip(range(1, W + 1), counts)
    ]

    def slope(v):
        pts = [(w + 1, math.log10(x)) for w, x in enumerate(v) if x > 0]
        if len(pts) < 2:
            return math.nan
        mw = ssum(p[0] for p in pts) / len(pts)
        ml = ssum(p[1] for p in pts) / len(pts)
        return ssum((p[0] - mw) * (p[1] - ml) for p in pts) / ssum((p[0] - mw) ** 2 for p in pts)

    return RichResult(
        payload={
            "order": order,
            "channel": [[1 if v else 0 for v in r] for r in order],
            "segments": segs,
            "counts": counts,
            "mean_lengths": mlen,
            "bifurcation_ratio": 10 ** -slope(counts),
            "length_ratio": 10 ** slope(mlen),
        }
    )


def channel_slope(distance, elevation) -> RichResult:
    r"""Channel slope of a long profile (distance upstream from the outlet, bed elevation): simple, 10-85, equal-area and harmonic.

    ``simple`` ``(z_L - z_0)/L``; ``s1085`` between 10 % and 85 % of the length
    (linear interpolation); ``equal_area`` ``2 A / L^2`` with ``A`` the
    trapezoidal area between the profile and the outlet elevation;
    ``harmonic`` (Taylor-Schwarz) ``[L / sum (l_i / sqrt S_i)]^2`` over reaches
    with positive slope.

    References
    ----------
    Taylor, A. B. and Schwarz, H. E. (1952). Unit-hydrograph lag and peak flow
    related to basin characteristics. *Transactions, AGU*, 33(2), 235-246.

    Examples
    --------
    >>> r = channel_slope([0, 100, 200], [10, 11, 14])
    >>> r.simple, r.equal_area
    (0.02, 0.015)
    """
    x = [float(v) for v in distance]
    z = [float(v) for v in elevation]
    L = x[-1] - x[0]

    def at(u):
        for i in range(len(x) - 1):
            if x[i] <= u <= x[i + 1]:
                return z[i] + (z[i + 1] - z[i]) * (u - x[i]) / (x[i + 1] - x[i])
        return z[-1]

    A = ssum(0.5 * ((z[i] - z[0]) + (z[i + 1] - z[0])) * (x[i + 1] - x[i]) for i in range(len(x) - 1))
    reach = [(x[i + 1] - x[i], (z[i + 1] - z[i]) / (x[i + 1] - x[i])) for i in range(len(x) - 1)]
    pos = [(li, si) for li, si in reach if si > 0]
    harm = (ssum(li for li, _ in pos) / ssum(li / math.sqrt(si) for li, si in pos)) ** 2 if pos else 0.0
    return RichResult(
        payload={
            "simple": (z[-1] - z[0]) / L,
            "s1085": (at(x[0] + 0.85 * L) - at(x[0] + 0.10 * L)) / (0.75 * L),
            "equal_area": 2 * A / L**2,
            "harmonic": harm,
            "length": L,
        }
    )


def meander_metrics(x, y) -> RichResult:
    r"""Meander geometry of a channel centreline: sinuosity, radius of curvature, inflection points and wavelength.

    Sinuosity = along-channel length / straight end-to-end distance; the
    signed curvature at each interior vertex is that of the circle through it
    and its two neighbours (``radius = 1/|curvature|``); inflections are sign
    changes of the curvature; the mean wavelength is twice the mean straight
    distance between successive inflection vertices (Leopold and Wolman
    1960).

    References
    ----------
    Leopold, L. B. and Wolman, M. G. (1960). River meanders. *GSA Bulletin*,
    71(6), 769-794.

    Examples
    --------
    >>> round(meander_metrics([0, 1, 2], [0, 1, 0]).sinuosity, 12)
    1.414213562373
    """
    P = list(zip((float(v) for v in x), (float(v) for v in y)))
    n = len(P)
    seg = [math.dist(P[i], P[i + 1]) for i in range(n - 1)]
    length = ssum(seg)
    curv = []
    for i in range(1, n - 1):
        (ax, ay), (bx, by), (cx, cy) = P[i - 1], P[i], P[i + 1]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        curv.append(2 * cr / (seg[i - 1] * seg[i] * math.dist(P[i - 1], P[i + 1])))
    infl = [i + 1 for i in range(len(curv) - 1) if curv[i] * curv[i + 1] < 0]
    wl = [2 * math.dist(P[a], P[b]) for a, b in zip(infl, infl[1:])]
    return RichResult(
        payload={
            "length": length,
            "sinuosity": length / math.dist(P[0], P[-1]),
            "curvature": curv,
            "radius": [1 / abs(c) if c else math.inf for c in curv],
            "inflections": infl,
            "wavelength": ssum(wl) / len(wl) if wl else math.nan,
        }
    )


def darcy_flow(head, K, *, res: float = 1.0, porosity: float | None = None) -> RichResult:
    r"""Darcy flux ``q = -K grad h`` over a grid of hydraulic heads (rows run north to south).

    Gradients by central differences (one-sided at the edges), ``x``
    eastward along columns and ``y`` northward (against the row index);
    ``K`` scalar or a grid. With effective ``porosity`` also the seepage
    (average linear) velocity ``|q| / n_e``.

    References
    ----------
    Freeze, R. A. and Cherry, J. A. (1979). *Groundwater*. Prentice-Hall.

    Examples
    --------
    >>> r = darcy_flow([[10, 9, 8], [10, 9, 8]], 2.0, res=10.0)
    >>> r.qx[0], r.qy[0]
    ([0.2, 0.2, 0.2], [0.0, 0.0, 0.0])
    """
    H = _grid(head)
    nr, nc = len(H), len(H[0])
    Kg = [[float(K)] * nc for _ in range(nr)] if not hasattr(K, "__len__") else _grid(K)

    def d(a, b, span):
        return (a - b) / (span * res)

    qx = [[0.0] * nc for _ in range(nr)]
    qy = [[0.0] * nc for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            if nc > 1:
                lo, hi = max(j - 1, 0), min(j + 1, nc - 1)
                qx[i][j] = -Kg[i][j] * d(H[i][hi], H[i][lo], hi - lo)
            if nr > 1:
                lo, hi = max(i - 1, 0), min(i + 1, nr - 1)
                qy[i][j] = -Kg[i][j] * -d(H[hi][j], H[lo][j], hi - lo)
    mag = [[math.hypot(qx[i][j], qy[i][j]) for j in range(nc)] for i in range(nr)]
    out = {"qx": qx, "qy": qy, "magnitude": mag}
    if porosity is not None:
        out["velocity"] = [[v / porosity for v in r] for r in mag]
    return RichResult(payload=out)


def breach_depressions(dem, *, epsilon: float = 0.0) -> list:
    r"""Remove depressions by carving (breaching) rather than filling: priority flood with path lowering (Soille 2004).

    Cells are processed from the grid edge (and NA cells) upward in a
    priority queue, remembering from which cell each was reached; when a
    cell is not higher than the cell it was reached from, the path from that
    cell back to the edge is lowered (each cell to at most its downstream
    neighbour's elevation minus ``epsilon``) instead of raising the cell.
    Elevations only decrease; every cell ends with a non-ascending path to
    the edge (strictly descending with ``epsilon > 0``).

    References
    ----------
    Soille, P. (2004). Optimal removal of spurious pits in grid digital
    elevation models. *Water Resources Research*, 40(12), W12509.
    Lindsay, J. B. (2016). Efficient hybrid breaching-filling sink removal
    methods for flow path enforcement in digital elevation models.
    *Hydrological Processes*, 30(6), 846-857.

    Examples
    --------
    >>> breach_depressions([[5, 5, 5, 5], [5, 1, 3, 5], [5, 5, 2, 5], [5, 5, 0, 5]])[2]
    [5.0, 5.0, 1.0, 5.0]
    """
    G = _grid(dem)
    nr, nc = len(G), len(G[0])
    Z = [r[:] for r in G]
    done = [[False] * nc for _ in range(nr)]
    parent = {}
    pq = []
    for i in range(nr):
        for j in range(nc):
            if not _ok(Z[i][j]):
                done[i][j] = True
                continue
            if (
                i in (0, nr - 1)
                or j in (0, nc - 1)
                or any(not _ok(G[i + a][j + b]) for _, (a, b) in _D8 if 0 <= i + a < nr and 0 <= j + b < nc)
            ):
                heapq.heappush(pq, (Z[i][j], i, j))
                done[i][j] = True
    while pq:
        _, i, j = heapq.heappop(pq)
        for _, (a, b) in _D8:
            x, y = i + a, j + b
            if 0 <= x < nr and 0 <= y < nc and not done[x][y]:
                done[x][y] = True
                parent[(x, y)] = (i, j)
                cz, p = Z[x][y], (i, j)
                while p is not None and Z[p[0]][p[1]] > cz - epsilon:
                    Z[p[0]][p[1]] = cz - epsilon
                    cz = Z[p[0]][p[1]]
                    p = parent.get(p)
                heapq.heappush(pq, (Z[x][y], x, y))
    return Z


def height_above_drainage(
    dem, flowdir, channel, *, res: float = 1.0, max_height: float | None = None, max_distance: float | None = None
) -> RichResult:
    r"""Height above nearest drainage (HAND; Renno et al. 2008) along D8 flow paths, with floodplain and riparian masks.

    Each cell follows its D8 path to the first channel cell: ``hand = z -
    z_drain`` and ``distance`` the flow-path length (``res`` or ``res sqrt 2``
    per step); cells whose path leaves the grid first get ``nan``. The
    ``floodplain`` mask is ``hand <= max_height`` and the ``riparian`` mask
    adds ``distance <= max_distance``.

    References
    ----------
    Renno, C. D. et al. (2008). HAND, a new terrain descriptor using SRTM-DEM:
    mapping terra-firme rainforest environments in Amazonia. *Remote Sensing
    of Environment*, 112(9), 3469-3481.
    Nobre, A. D. et al. (2011). Height above the nearest drainage - a
    hydrologically relevant new terrain model. *Journal of Hydrology*,
    404(1-2), 13-29.

    Examples
    --------
    >>> r = height_above_drainage([[3, 2, 1]], [[1, 1, 0]], [[0, 0, 1]])
    >>> r.hand, r.distance
    ([[2.0, 1.0, 0.0]], [[2.0, 1.0, 0.0]])
    """
    Z = _grid(dem)
    nr, nc = len(Z), len(Z[0])
    rec = _receivers(flowdir)
    hand = [[math.nan] * nc for _ in range(nr)]
    dist = [[math.nan] * nc for _ in range(nr)]
    drain = {}

    def resolve(c):
        path = []
        while c not in drain:
            if channel[c[0]][c[1]]:
                drain[c] = (c, 0.0)
                break
            path.append(c)
            d = rec.get(c)
            if d is None or d in path:
                drain[c] = (None, math.nan)
                break
            c = d
        for p in reversed(path):
            d = rec.get(p)
            tgt, L = drain.get(d, (None, math.nan)) if d is not None else (None, math.nan)
            if tgt is None:
                drain[p] = (None, math.nan)
            else:
                a, b = _OFF[flowdir[p[0]][p[1]]]
                drain[p] = (tgt, L + res * (math.sqrt(2.0) if a and b else 1.0))

    for i in range(nr):
        for j in range(nc):
            if _ok(Z[i][j]):
                resolve((i, j))
                tgt, L = drain[(i, j)]
                if tgt is not None:
                    hand[i][j] = Z[i][j] - Z[tgt[0]][tgt[1]]
                    dist[i][j] = L
    out = {"hand": hand, "distance": dist}
    if max_height is not None:
        out["floodplain"] = [[1 if v == v and v <= max_height else 0 for v in r] for r in hand]
        if max_distance is not None:
            out["riparian"] = [
                [
                    1 if hand[i][j] == hand[i][j] and hand[i][j] <= max_height and dist[i][j] <= max_distance else 0
                    for j in range(nc)
                ]
                for i in range(nr)
            ]
    return RichResult(payload=out)


def cheatsheet() -> str:
    return (
        "flood_frequency / low_flow_frequency / flow_duration_curve / idf_fit / unit_hydrograph / convolve_runoff / "
        "baseflow_filter / hydrograph_summary / stream_segments / channel_slope / meander_metrics / darcy_flow / "
        "breach_depressions / height_above_drainage -> hydrology."
    )

# alias kept from the retired placeholder of the same name
flow_duration = flow_duration_curve
