# morie.fn -- function file (rootcoder007/morie)
"""Monte Carlo estimation over spatial domains: integration and means over boxes or polygons (plain, antithetic,
Latin hypercube), stratified sampling with proportional or Neyman allocation, hit-or-miss zonal estimation,
control variates and convergence diagnostics."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["mc_integrate", "mc_stratified", "mc_zonal", "mc_control_variate", "mc_convergence"]


def _poly(p):
    return [(float(a), float(b)) for a, b in p]


def _in_poly(pt, polygon) -> bool:
    """Ray-casting point-in-polygon test (even-odd rule)."""
    x, y = float(pt[0]), float(pt[1])
    P = _poly(polygon)
    inside = False
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _area(P):
    n = len(P)
    return abs(ssum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))) / 2


def _points(n, bounds, seed, stream, method):
    d = len(bounds)
    if method == "lhs":
        cols = []
        for j in range(d):
            u = [float(v) for v in random_uniform(2 * n, seed=seed, stream=stream + j)]
            perm = list(range(n))
            for t in range(n):  # Fisher-Yates with the first n uniforms
                k = t + int(u[t] * (n - t))
                perm[t], perm[k] = perm[k], perm[t]
            cols.append([(perm[i] + u[n + i]) / n for i in range(n)])
        U = [[cols[j][i] for j in range(d)] for i in range(n)]
    else:
        m = n // 2 if method == "antithetic" else n
        flat = [float(v) for v in random_uniform(m * d, seed=seed, stream=stream)]
        U = [flat[i * d : (i + 1) * d] for i in range(m)]
        if method == "antithetic":
            U = U + [[1 - v for v in r] for r in U]
    return [[lo + u * (hi - lo) for u, (lo, hi) in zip(r, bounds)] for r in U]


def mc_integrate(f, *, bounds=None, polygon=None, n: int = 10000, seed: int = 1, method: str = "plain") -> RichResult:
    r"""Monte Carlo integral and mean of ``f`` over a box (``bounds``) or a polygon (rejection from its bounding box).

    ``I ~ |A| mean f(X_i)`` with ``X_i`` uniform on the domain; ``method``
    ``plain`` (Philox uniforms, stream 0), ``antithetic`` (pairs ``u``,
    ``1 - u``; the standard error uses the pair averages) or ``lhs`` (Latin
    hypercube, one stratum per point in each coordinate; McKay, Beckman and
    Conover 1979). For polygons, box points outside are discarded and ``|A|``
    is the polygon area; the result reports the domain mean, integral,
    standard error of the mean and the number of points used.

    References
    ----------
    McKay, M. D., Beckman, R. J. and Conover, W. J. (1979). A comparison of
    three methods for selecting values of input variables in the analysis of
    output from a computer code. *Technometrics*, 21(2), 239-245.

    Examples
    --------
    >>> r = mc_integrate(lambda x, y: x + y, bounds=[(0, 1), (0, 1)], n=2000, method="lhs")
    >>> round(r.integral, 2)
    1.0
    """
    if polygon is not None:
        P = _poly(polygon)
        bounds = [(min(p[0] for p in P), max(p[0] for p in P)), (min(p[1] for p in P), max(p[1] for p in P))]
        area = _area(P)
    else:
        area = math.prod(hi - lo for lo, hi in bounds)
    X = _points(n, [(float(a), float(b)) for a, b in bounds], seed, 0, method)
    if polygon is not None:
        X = [x for x in X if _in_poly(x, P)]
    vals = [float(f(*x)) for x in X]
    m = len(vals)
    mean = ssum(vals) / m
    if method == "antithetic" and polygon is None:
        h = m // 2
        pairs = [(vals[i] + vals[h + i]) / 2 for i in range(h)]
        se = math.sqrt(ssum((v - mean) ** 2 for v in pairs) / (h - 1) / h)
    else:
        se = math.sqrt(ssum((v - mean) ** 2 for v in vals) / (m - 1) / m)
    return RichResult(
        payload={
            "mean": mean,
            "integral": mean * area,
            "se_mean": se,
            "se_integral": se * area,
            "n_used": m,
            "area": area,
        }
    )


def mc_stratified(
    f, strata, *, n: int = 1000, allocation: str = "proportional", pilot_sd=None, seed: int = 1
) -> RichResult:
    r"""Stratified Monte Carlo estimate of the domain mean of ``f`` over polygon (or box) strata.

    Stratum ``h`` (weight ``W_h`` = area share) gets ``n_h`` points:
    ``proportional`` ``n W_h`` or ``neyman`` ``n W_h S_h / sum W S`` from
    ``pilot_sd`` (Cochran 1977, section 5.5), at least 2 each. The estimate
    is ``sum W_h ybar_h`` with variance ``sum W_h^2 s_h^2 / n_h``. Stratum
    ``h`` uses Philox stream ``h``.

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques*, 3rd ed. Wiley, chapter 5.

    Examples
    --------
    >>> sq = [[(0, 0), (0.5, 0), (0.5, 1), (0, 1)], [(0.5, 0), (1, 0), (1, 1), (0.5, 1)]]
    >>> r = mc_stratified(lambda x, y: 1.0 if x < 0.5 else 3.0, sq, n=100)
    >>> r.mean, r.se
    (2.0, 0.0)
    """
    polys = [_poly(s) for s in strata]
    areas = [_area(p) for p in polys]
    W = [a / ssum(areas) for a in areas]
    if allocation == "proportional":
        nh = [max(2, round(n * w)) for w in W]
    elif allocation == "neyman":
        S = [float(v) for v in pilot_sd]
        tot = ssum(w * s for w, s in zip(W, S))
        nh = [max(2, round(n * w * s / tot)) for w, s in zip(W, S)]
    else:
        raise ValueError("allocation must be proportional or neyman")
    means, vars_ = [], []
    for h, (p, k) in enumerate(zip(polys, nh)):
        bx = [(min(q[0] for q in p), max(q[0] for q in p)), (min(q[1] for q in p), max(q[1] for q in p))]
        got, stream = [], 0
        while len(got) < k:
            cand = _points(4 * k, bx, seed, 1000 * h + stream, "plain")
            got += [c for c in cand if _in_poly(c, p)]
            stream += 1
        vals = [float(f(*x)) for x in got[:k]]
        m = ssum(vals) / k
        means.append(m)
        vars_.append(ssum((v - m) ** 2 for v in vals) / (k - 1))
    est = ssum(w * m for w, m in zip(W, means))
    var = ssum(w * w * v / k for w, v, k in zip(W, vars_, nh))
    return RichResult(payload={"mean": est, "se": math.sqrt(var), "n_h": nh, "stratum_means": means, "weights": W})


def mc_zonal(indicator, bounds, *, value=None, n: int = 10000, seed: int = 1) -> RichResult:
    r"""Hit-or-miss Monte Carlo area of a zone ``{x: indicator(x)}`` inside a box, and the zone mean of ``value``.

    Area ``|B| p_hat`` with binomial standard error ``|B| sqrt(p_hat (1 -
    p_hat) / n)``.

    Examples
    --------
    >>> r = mc_zonal(lambda x, y: x * x + y * y <= 1, [(-1, 1), (-1, 1)], n=20000)
    >>> abs(r.area - math.pi) < 4 * r.se_area
    True
    """
    B = [(float(a), float(b)) for a, b in bounds]
    X = _points(n, B, seed, 0, "plain")
    hits = [x for x in X if indicator(*x)]
    p = len(hits) / n
    box = math.prod(b - a for a, b in B)
    out = {"area": box * p, "se_area": box * math.sqrt(p * (1 - p) / n), "hits": len(hits)}
    if value is not None and hits:
        vals = [float(value(*x)) for x in hits]
        m = ssum(vals) / len(vals)
        out["zone_mean"] = m
        out["zone_mean_se"] = (
            math.sqrt(ssum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) if len(vals) > 1 else float("nan")
        )
    return RichResult(payload=out)


def mc_control_variate(y, g, g_mean: float) -> RichResult:
    r"""Control-variate estimate ``ybar - b (gbar - E g)`` with the estimated optimal ``b = cov(y, g)/var(g)``.

    Returns the estimate, its standard error ``sqrt(s_y^2 (1 - r^2) / n)`` and
    the variance reduction factor ``1 - r^2`` (Glasserman 2004, section 4.1).

    References
    ----------
    Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
    Springer.

    Examples
    --------
    >>> r = mc_control_variate([1.0, 2.0, 3.0, 4.0], [0.9, 2.1, 2.9, 4.1], 2.5)
    >>> round(r.estimate, 6)
    2.5
    """
    Y, G = [float(v) for v in y], [float(v) for v in g]
    n = len(Y)
    my, mg = ssum(Y) / n, ssum(G) / n
    syy = ssum((a - my) ** 2 for a in Y) / (n - 1)
    sgg = ssum((a - mg) ** 2 for a in G) / (n - 1)
    syg = ssum((a - my) * (b - mg) for a, b in zip(Y, G)) / (n - 1)
    b = syg / sgg
    r2 = syg * syg / (syy * sgg)
    return RichResult(
        payload={
            "estimate": my - b * (mg - g_mean),
            "b": b,
            "se": math.sqrt(syy * (1 - r2) / n),
            "variance_reduction": 1 - r2,
        }
    )


def mc_convergence(samples, *, batches: int = 20) -> RichResult:
    r"""Convergence diagnostics of a Monte Carlo sample: running mean and 95% band, and the batch-means MCSE.

    The batch-means standard error splits the sample into ``batches``
    consecutive batches and uses ``sd(batch means)/sqrt(batches)``; its ratio
    to the naive iid standard error is an inefficiency factor, and ``n /
    factor^2`` an effective sample size (Flegal, Haran and Jones 2008).

    References
    ----------
    Flegal, J. M., Haran, M. and Jones, G. L. (2008). Markov chain Monte
    Carlo: can we trust the third significant figure? *Statistical Science*,
    23(2), 250-260.

    Examples
    --------
    >>> r = mc_convergence([1.0, 3.0, 2.0, 4.0], batches=2)
    >>> r.running_mean, r.batch_se
    ([1.0, 2.0, 2.0, 2.5], 0.5)
    """
    x = [float(v) for v in samples]
    n = len(x)
    run, s, lo, hi = [], 0.0, [], []
    for i, v in enumerate(x, 1):
        s += v
        run.append(s / i)
    for i in range(1, n + 1):
        seg = x[:i]
        m = run[i - 1]
        sd = math.sqrt(ssum((v - m) ** 2 for v in seg) / (i - 1)) if i > 1 else float("nan")
        lo.append(m - 1.96 * sd / math.sqrt(i))
        hi.append(m + 1.96 * sd / math.sqrt(i))
    b = n // batches
    bm = [ssum(x[k * b : (k + 1) * b]) / b for k in range(batches)]
    mb = ssum(bm) / batches
    bse = math.sqrt(ssum((v - mb) ** 2 for v in bm) / (batches - 1)) / math.sqrt(batches)
    mean = run[-1]
    iid = math.sqrt(ssum((v - mean) ** 2 for v in x) / (n - 1)) / math.sqrt(n)
    fac = bse / iid if iid > 0 else float("nan")
    return RichResult(
        payload={
            "running_mean": run,
            "lower": lo,
            "upper": hi,
            "batch_se": bse,
            "iid_se": iid,
            "inefficiency": fac,
            "ess": n / fac**2 if fac and fac == fac else float("nan"),
        }
    )


def cheatsheet() -> str:
    return "mc_integrate / mc_stratified / mc_zonal / mc_control_variate / mc_convergence -> spatial Monte Carlo."
