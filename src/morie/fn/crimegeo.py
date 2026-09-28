# morie.fn -- function file (rootcoder007/morie)
"""Crime geography for police analysts: CrimeStat journey-to-crime distance-decay functions, likelihood
surfaces for a serial offender's residence and their calibration from known offender trips, Canter's
circle hypothesis, search-cost (hit score) evaluation, and risk terrain modelling (proximity and density
risk layers, Poisson regression, relative risk values and the composite risk surface)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = [
    "jtc_decay",
    "jtc_surface",
    "jtc_calibrate",
    "circle_hypothesis",
    "search_cost",
    "risk_layers",
    "risk_terrain",
]

_DEFAULTS = {  # Levine (2013), CrimeStat IV chapter 13, Baltimore County calibration
    "linear": {"A": 1.9, "B": -0.06},
    "negative_exponential": {"A": 1.89, "B": 0.06},
    "normal": {"A": 29.5, "mean": 4.2, "sd": 4.6},
    "lognormal": {"A": 8.6, "mean": 4.2, "sd": 4.6},
    "truncated_negative_exponential": {"cutoff": 0.4, "peak": 13.8, "C": 0.2},
}


def _decay1(d, function, p):
    if function == "linear":
        return max(0.0, p["A"] + p["B"] * d)
    if function == "negative_exponential":
        return p["A"] * math.exp(-p["B"] * d)
    if function == "normal":
        z = (d - p["mean"]) / p["sd"]
        return p["A"] / (p["sd"] * math.sqrt(2 * math.pi)) * math.exp(-z * z / 2)
    if function == "lognormal":
        if d <= 0:
            return 0.0
        L = math.log(d * d)
        return (
            p["A"] / (d * d * p["sd"] * math.sqrt(2 * math.pi)) * math.exp(-((L - p["mean"]) ** 2) / (2 * p["sd"] ** 2))
        )
    if function == "truncated_negative_exponential":
        if d <= p["cutoff"]:
            return p["peak"] / p["cutoff"] * d
        return p["peak"] * math.exp(-p["C"] * (d - p["cutoff"]))
    raise ValueError(
        "function must be linear, negative_exponential, normal, lognormal or truncated_negative_exponential"
    )


def _params(function, params):
    if function not in _DEFAULTS:
        _decay1(0.0, function, {})
    p = dict(_DEFAULTS[function])
    p.update({k: float(v) for k, v in (params or {}).items()})
    return p


def jtc_decay(d, function: str = "negative_exponential", params=None) -> list:
    r"""CrimeStat journey-to-crime distance-decay likelihood ``f(d)`` (Levine 2013, equations 13.14-13.20).

    ``linear`` ``max(0, A + B d)``; ``negative_exponential`` ``A e^{-B d}``
    (``B > 0`` is the decay rate); ``normal`` ``A phi((d - mean)/sd) / sd``;
    ``lognormal`` ``A / (d^2 sd sqrt(2 pi)) exp(-(ln d^2 - mean)^2 / (2
    sd^2))`` (equations 13.18 and 13.32-13.36; 0 at ``d = 0``);
    ``truncated_negative_exponential`` ``(peak / cutoff) d`` up to
    ``cutoff`` and ``peak e^{-C (d - cutoff)}`` beyond. ``params`` overrides
    the Baltimore County defaults of CrimeStat (distances in miles).

    References
    ----------
    Levine, N. (2013). Journey-to-crime estimation. Chapter 13 in
    *CrimeStat IV: A Spatial Statistics Program for the Analysis of Crime
    Incident Locations*, version 4.0. Ned Levine & Associates and the
    National Institute of Justice (NCJ 242973).

    Examples
    --------
    >>> [round(v, 6) for v in jtc_decay([0.0, 0.4, 1.4], "truncated_negative_exponential")]
    [0.0, 13.8, 11.298484]
    >>> jtc_decay([40.0], "linear")
    [0.0]
    """
    p = _params(function, params)
    D = [d] if isinstance(d, (int, float)) else list(d)
    return [_decay1(float(v), function, p) for v in D]


def _dist(a, b, metric):
    if metric == "euclidean":
        return math.hypot(a[0] - b[0], a[1] - b[1])
    if metric == "manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    raise ValueError("metric must be euclidean or manhattan")


def jtc_surface(
    incidents, grid, function: str = "negative_exponential", params=None, *, metric: str = "euclidean"
) -> RichResult:
    r"""Journey-to-crime likelihood surface of a serial offender's residence over ``grid`` points.

    The score of grid point ``g`` is ``sum_n f(d(g, x_n))`` over the
    incidents, ``f`` a :func:`jtc_decay` function and ``d`` Euclidean
    (direct) or Manhattan (indirect) distance (Levine 2013). Returns the
    scores, their normalised version (summing to 1) and the peak point.

    Examples
    --------
    >>> r = jtc_surface([(0, 0), (2, 0)], [(1, 0), (5, 0)], "linear", {"A": 1.0, "B": -0.1})
    >>> [round(v, 6) for v in r.score], r.peak
    ([1.8, 1.2], (1.0, 0.0))
    """
    p = _params(function, params)
    X = [(float(a), float(b)) for a, b in incidents]
    G = [(float(a), float(b)) for a, b in grid]
    s = [ssum(_decay1(_dist(g, x, metric), function, p) for x in X) for g in G]
    tot = ssum(s)
    k = max(range(len(G)), key=lambda i: (s[i], -i))
    return RichResult(
        payload={
            "score": s,
            "probability": [v / tot for v in s] if tot > 0 else [math.nan] * len(s),
            "peak": G[k],
            "peak_index": k,
        }
    )


def _ols(x, y, intercept=True):
    n = len(x)
    if not intercept:
        return 0.0, ssum(a * b for a, b in zip(x, y)) / ssum(a * a for a in x)
    mx, my = ssum(x) / n, ssum(y) / n
    b = ssum((a - mx) * (c - my) for a, c in zip(x, y)) / ssum((a - mx) ** 2 for a in x)
    return my - b * mx, b


def jtc_calibrate(distances, breaks) -> RichResult:
    r"""Calibrate the five CrimeStat decay functions from known offender journey distances (Levine 2013).

    Distances are grouped by ``breaks``; ``pct_i`` is the percentage in bin
    ``i`` and ``d_i`` its midpoint. ``linear``: OLS ``pct = A + B d``
    (13.22); ``negative_exponential``: OLS ``ln pct = ln A - B d`` with
    ``ln 0`` taken as -16 (13.24-13.26); ``normal`` and ``lognormal``:
    ``mean`` and ``sd`` of the raw distances, then ``pct`` regressed without
    intercept on the normal (13.27-13.30) or lognormal (13.31-13.37) kernel
    for ``A``; ``truncated_negative_exponential``: ``cutoff`` the midpoint of
    the modal bin, ``peak`` its percentage, and ``C`` from OLS of ``ln pct``
    on ``d`` beyond the cutoff (13.38-13.44). Also returns the residual sum of
    squares of each fitted function on the bins.

    Examples
    --------
    >>> r = jtc_calibrate([0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5, 5.5], [0, 1, 2, 3, 4, 5, 6])
    >>> r.pct, r.params["truncated_negative_exponential"]["cutoff"]
    ([10.0, 20.0, 30.0, 20.0, 10.0, 10.0], 2.5)
    """
    D = [float(v) for v in distances]
    B = [float(v) for v in breaks]
    n = len(D)
    cnt = [0] * (len(B) - 1)
    for v in D:
        for i in range(len(B) - 1):
            if B[i] <= v < B[i + 1] or (i == len(B) - 2 and v == B[-1]):
                cnt[i] += 1
                break
    pct = [100 * c / n for c in cnt]
    mid = [(B[i] + B[i + 1]) / 2 for i in range(len(B) - 1)]
    lp = [math.log(v) if v > 0 else -16.0 for v in pct]
    out = {}
    a, b = _ols(mid, pct)
    out["linear"] = {"A": a, "B": b}
    k, b = _ols(mid, lp)
    out["negative_exponential"] = {"A": math.exp(k), "B": -b}
    m = ssum(D) / n
    sd = math.sqrt(ssum((v - m) ** 2 for v in D) / (n - 1))
    for fn in ("normal", "lognormal"):
        ker = [_decay1(d, fn, {"A": 1.0, "mean": m, "sd": sd}) for d in mid]
        out[fn] = {"A": _ols(ker, pct, intercept=False)[1], "mean": m, "sd": sd}
    j = max(range(len(pct)), key=lambda i: (pct[i], -i))
    beyond = [i for i in range(len(mid)) if mid[i] > mid[j]]
    C = -_ols([mid[i] for i in beyond], [lp[i] for i in beyond])[1] if len(beyond) >= 2 else math.nan
    out["truncated_negative_exponential"] = {"cutoff": mid[j], "peak": pct[j], "C": C}
    rss = {fn: ssum((pct[i] - _decay1(mid[i], fn, _params(fn, out[fn]))) ** 2 for i in range(len(mid))) for fn in out}
    return RichResult(
        payload={"pct": pct, "midpoints": mid, "params": out, "rss": rss, "best": min(rss, key=lambda f: rss[f])}
    )


def circle_hypothesis(crimes, home=None) -> RichResult:
    r"""Canter and Larkin's circle hypothesis for a serial offender's crime locations.

    The circle has as diameter the segment joining the two crimes farthest
    apart; ``marauder`` is ``True`` when the offender's ``home`` lies inside
    it (``None`` when no home is given), otherwise the offender is classed as
    a commuter. Also returns the share of crimes inside the circle.

    References
    ----------
    Canter, D. and Larkin, P. (1993). The environmental range of serial
    rapists. *Journal of Environmental Psychology*, 13(1), 63-69.

    Examples
    --------
    >>> r = circle_hypothesis([(0, 0), (4, 0), (2, 1)], home=(2, -1))
    >>> r.center, r.radius, r.marauder, round(r.share_inside, 6)
    ((2.0, 0.0), 2.0, True, 1.0)
    """
    P = [(float(a), float(b)) for a, b in crimes]
    n = len(P)
    i, j = max(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda t: math.dist(P[t[0]], P[t[1]]))
    c = ((P[i][0] + P[j][0]) / 2, (P[i][1] + P[j][1]) / 2)
    r = math.dist(P[i], P[j]) / 2
    tol = 1e-12 * max(1.0, r)
    inside = sum(1 for p in P if math.dist(p, c) <= r + tol) / n
    mar = None if home is None else math.dist((float(home[0]), float(home[1])), c) <= r + tol
    return RichResult(payload={"center": c, "radius": r, "pair": (i, j), "marauder": mar, "share_inside": inside})


def search_cost(scores, home_index: int) -> RichResult:
    r"""Hit score of a geographic profile: the share of the search area searched before the offender's home.

    ``hit_percent = 100 #{j: s_j > s_home} / n`` (cells strictly ranked
    above the home cell), with ``ties`` cells equal to it; the
    ``hit_percent_ties`` variant counts ties as searched (Rossmo 2000).

    References
    ----------
    Rossmo, D. K. (2000). *Geographic Profiling*. CRC Press.

    Examples
    --------
    >>> search_cost([5.0, 3.0, 9.0, 3.0], 1).hit_percent
    50.0
    """
    s = [float(v) for v in scores]
    h = s[home_index]
    above = sum(1 for v in s if v > h)
    ties = sum(1 for v in s if v == h)
    return RichResult(
        payload={
            "hit_percent": 100 * above / len(s),
            "hit_percent_ties": 100 * (above + ties) / len(s),
            "rank": above + 1,
            "ties": ties,
        }
    )


def risk_layers(cells, features, *, radius: float, kind: str = "proximity") -> list:
    r"""One risk-terrain layer: for each cell centre, ``proximity`` (1 if a feature is within ``radius``, else 0)
    or ``density`` (the number of features within ``radius``) (Caplan, Kennedy and Miller 2011).

    Examples
    --------
    >>> risk_layers([(0, 0), (5, 0)], [(1, 0), (1.5, 0)], radius=2)
    [1, 0]
    >>> risk_layers([(0, 0), (5, 0)], [(1, 0), (1.5, 0)], radius=2, kind="density")
    [2, 0]
    """
    C = [(float(a), float(b)) for a, b in cells]
    F = [(float(a), float(b)) for a, b in features]
    cnt = [sum(1 for f in F if math.dist(c, f) <= radius) for c in C]
    if kind == "proximity":
        return [1 if v else 0 for v in cnt]
    if kind == "density":
        return cnt
    raise ValueError("kind must be proximity or density")


def _solve(A, b):
    n = len(b)
    M = [list(r) + [v] for r, v in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _inverse(A):
    n = len(A)
    return [list(col) for col in zip(*[_solve(A, [float(i == j) for i in range(n)]) for j in range(n)])]


def risk_terrain(counts, layers, *, names=None, tol: float = 1e-10, maxit: int = 50) -> RichResult:
    r"""Risk terrain model: Poisson regression of cell crime counts on risk layers, relative risk values and risk.

    ``log E(y_c) = b0 + sum_k b_k x_ck`` fitted by iteratively reweighted
    least squares (the MLE of ``glm(family = poisson)``). Each layer's
    relative risk value is ``RRV_k = exp(b_k)``; the composite relative risk
    of a cell is ``exp(sum_k b_k x_ck)`` (the product of the RRVs of the
    risk factors present in a binary model), reported with its version
    rescaled so the lowest-risk cell is 1 (Caplan, Kennedy and Miller 2011).
    Standard errors come from the inverse Fisher information.

    References
    ----------
    Caplan, J. M., Kennedy, L. W. and Miller, J. (2011). Risk terrain
    modeling: brokering criminological theory and GIS methods for crime
    forecasting. *Justice Quarterly*, 28(2), 360-381.

    Examples
    --------
    >>> r = risk_terrain([1, 2, 4, 8], [[0, 1, 0, 1], [0, 0, 1, 1]])
    >>> [round(v, 6) for v in r.rrv]
    [2.0, 4.0]
    """
    y = [float(v) for v in counts]
    L = [[float(v) for v in col] for col in layers]
    n, p = len(y), len(L) + 1
    X = [[1.0] + [L[k][i] for k in range(len(L))] for i in range(n)]
    mu0 = ssum(y) / n
    beta = [math.log(mu0)] + [0.0] * (p - 1)
    dev_old = math.inf
    for _ in range(maxit):
        eta = [ssum(b * x for b, x in zip(beta, r)) for r in X]
        mu = [math.exp(e) for e in eta]
        z = [e + (yy - m) / m for e, yy, m in zip(eta, y, mu)]
        XtWX = [[ssum(X[i][a] * mu[i] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        XtWz = [ssum(X[i][a] * mu[i] * z[i] for i in range(n)) for a in range(p)]
        beta = _solve(XtWX, XtWz)
        mu = [math.exp(ssum(b * x for b, x in zip(beta, r))) for r in X]
        dev = 2 * ssum((yy * math.log(yy / m) if yy > 0 else 0.0) - (yy - m) for yy, m in zip(y, mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev
    info = [[ssum(X[i][a] * mu[i] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    se = [math.sqrt(v) for v in (row[i] for i, row in enumerate(_inverse(info)))]
    rel = [math.exp(ssum(b * x for b, x in zip(beta[1:], r[1:]))) for r in X]
    lo = min(rel)
    zs = [b / s for b, s in zip(beta, se)]
    return RichResult(
        payload={
            "names": list(names) if names else [f"layer{k + 1}" for k in range(p - 1)],
            "coefficients": beta,
            "se": se,
            "z": zs,
            "pvalue": [2 * (1 - float(pnorm(abs(v)))) for v in zs],
            "rrv": [math.exp(b) for b in beta[1:]],
            "relative_risk": rel,
            "risk_score": [v / lo for v in rel],
            "fitted": mu,
            "deviance": dev,
        }
    )


def cheatsheet() -> str:
    return (
        "jtc_decay / jtc_surface / jtc_calibrate / circle_hypothesis / search_cost / risk_layers / risk_terrain "
        "-> CrimeStat journey-to-crime, Canter circle, hit score and risk terrain modelling."
    )
