# morie.fn -- function file (rootcoder007/morie)
"""Species distribution and home-range methods: MaxEnt (the exponential-family / Gibbs density
over background points with linear, quadratic, product and hinge features and ridge or lasso
regularisation, logistic and cloglog outputs) and k-LoCoH home ranges with exact union areas."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["maxent_features", "maxent_fit", "maxent_predict", "locoh_home_range"]


def maxent_features(X, *, classes: str = "lq", hinge_knots: int = 5, ranges=None) -> RichResult:
    r"""MaxEnt feature expansion of environmental covariates (Phillips et al. 2006, 2017).

    ``classes`` combines ``l`` (linear), ``q`` (squares), ``p`` (pairwise
    products) and ``h`` (hinges ``max(0, x - k)`` and ``max(0, k - x)`` at
    ``hinge_knots`` equally spaced knots). Linear, quadratic and product
    features are scaled to [0, 1] over ``ranges`` (per-covariate min/max,
    default from ``X``); hinges are scaled by the range.

    References
    ----------
    Phillips, S. J., Anderson, R. P. and Schapire, R. E. (2006). Maximum
    entropy modeling of species geographic distributions. Ecol. Modelling
    190, 231-259. Phillips, S. J. et al. (2017). Opening the black box: an
    open-source release of Maxent. Ecography 40, 887-893.

    Examples
    --------
    >>> r = maxent_features([[0.0], [1.0], [2.0]], classes="lq")
    >>> r.features
    [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]]
    """
    Xs = [[float(v) for v in r] for r in X]
    p = len(Xs[0])
    rng = ranges if ranges is not None else [(min(r[j] for r in Xs), max(r[j] for r in Xs)) for j in range(p)]

    def sc(v, j):
        lo, hi = rng[j]
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    names = []
    if "l" in classes:
        names += [("l", j) for j in range(p)]
    if "q" in classes:
        names += [("q", j) for j in range(p)]
    if "p" in classes:
        names += [("p", j, k) for j in range(p) for k in range(j + 1, p)]
    if "h" in classes:
        for j in range(p):
            lo, hi = rng[j]
            for m in range(1, hinge_knots + 1):
                kn = lo + (hi - lo) * m / (hinge_knots + 1)
                names += [("hf", j, kn), ("hr", j, kn)]

    def feat(row):
        out = []
        for nm in names:
            if nm[0] == "l":
                out.append(sc(row[nm[1]], nm[1]))
            elif nm[0] == "q":
                out.append(sc(row[nm[1]], nm[1]) ** 2)
            elif nm[0] == "p":
                out.append(sc(row[nm[1]], nm[1]) * sc(row[nm[2]], nm[2]))
            else:
                lo, hi = rng[nm[1]]
                span = hi - lo if hi > lo else 1.0
                d = (row[nm[1]] - nm[2]) / span
                out.append(max(0.0, d) if nm[0] == "hf" else max(0.0, -d))
        return out

    return RichResult(
        payload={
            "features": [feat(r) for r in Xs],
            "names": [list(nm) for nm in names],
            "ranges": [list(r) for r in rng],
        }
    )


def maxent_fit(
    presence,
    background,
    *,
    l2: float = 0.0,
    l1: float = 0.0,
    add_presence: bool = True,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> RichResult:
    r"""MaxEnt: the Gibbs distribution over background points maximising penalised presence likelihood.

    ``q(x) = exp(eta(x)) / sum_b exp(eta(b))``, ``eta = f(x)' lambda``, over
    the background feature rows (presence rows appended when
    ``add_presence``, as maxnet); ``lambda`` minimises
    ``-mean_presence eta + log sum_b exp(eta(b)) + (l2/2)||lambda||^2 + l1 ||lambda||_1``
    - Newton steps when ``l1 = 0`` (the maximum-entropy dual: model feature
    means match the presence means), otherwise cyclic coordinate descent with
    soft-thresholded Newton coordinate steps (the sequential-update MaxEnt of
    Dudik, Phillips and Schapire 2004). ``entropy`` is that of ``q``.

    References
    ----------
    Phillips, S. J., Anderson, R. P. and Schapire, R. E. (2006). Ecol.
    Modelling 190, 231-259. Dudik, M., Phillips, S. J. and Schapire, R. E.
    (2004). Performance guarantees for regularized maximum entropy density
    estimation. COLT 2004, 472-486.

    Examples
    --------
    >>> r = maxent_fit([[1.0], [0.8]], [[0.0], [0.5], [1.0]], add_presence=False)
    >>> round(r.lambdas[0], 8)
    3.44751481
    """
    P = [[float(v) for v in r] for r in presence]
    B = [[float(v) for v in r] for r in background] + (P if add_presence else [])
    k = len(P[0])
    nb = len(B)
    target = [ssum(r[j] for r in P) / len(P) for j in range(k)]
    lam = [0.0] * k

    def dist(lm):
        eta = [ssum(r[j] * lm[j] for j in range(k)) for r in B]
        mx = max(eta)
        w = [math.exp(v - mx) for v in eta]
        z = ssum(w)
        q = [v / z for v in w]
        return q, mx + math.log(z)

    def obj(lm, lz):
        return (
            -ssum(target[j] * lm[j] for j in range(k))
            + lz
            + 0.5 * l2 * ssum(v * v for v in lm)
            + l1 * ssum(abs(v) for v in lm)
        )

    q, lz = dist(lam)
    f = obj(lam, lz)
    it = 0
    for _ in range(max_iter):
        it += 1
        mu = [ssum(q[i] * B[i][j] for i in range(nb)) for j in range(k)]
        if l1 == 0.0:
            g = [mu[j] - target[j] + l2 * lam[j] for j in range(k)]
            if max(abs(v) for v in g) <= tol:
                break
            H = [
                [
                    ssum(q[i] * (B[i][a] - mu[a]) * (B[i][b] - mu[b]) for i in range(nb)) + (l2 if a == b else 0.0)
                    for b in range(k)
                ]
                for a in range(k)
            ]
            d = solve(H, [-v for v in g])
            step = 1.0
            while True:
                new = [a + step * b for a, b in zip(lam, d)]
                qn, lzn = dist(new)
                fn = obj(new, lzn)
                if fn <= f + 1e-4 * step * ssum(a * b for a, b in zip(g, d)) or step < 1e-10:
                    break
                step /= 2.0
            lam, q, lz, f = new, qn, lzn, fn
        else:
            change = 0.0
            for j in range(k):
                muj = ssum(q[i] * B[i][j] for i in range(nb))
                var = ssum(q[i] * (B[i][j] - muj) ** 2 for i in range(nb))
                if var + l2 <= 0:
                    continue
                # soft-thresholded Newton step on coordinate j of the local quadratic model
                z = lam[j] * var - (muj - target[j])
                newj = math.copysign(max(abs(z) - l1, 0.0), z) / (var + l2)
                if newj != lam[j]:
                    trial = list(lam)
                    step = 1.0
                    while True:
                        trial[j] = lam[j] + step * (newj - lam[j])
                        qn, lzn = dist(trial)
                        fn = obj(trial, lzn)
                        if fn <= f + 1e-13 * abs(f) or step < 1e-10:
                            break
                        step /= 2.0
                    change = max(change, abs(trial[j] - lam[j]))
                    lam, q, lz, f = trial, qn, lzn, fn
            if change <= tol:
                break
    ent = -ssum(v * math.log(v) for v in q if v > 0)
    return RichResult(
        payload={
            "lambdas": lam,
            "entropy": ent,
            "log_normaliser": lz,
            "objective": f,
            "iterations": it,
            "background": B,
        }
    )


def maxent_predict(fit, features, *, output: str = "cloglog") -> list:
    r"""MaxEnt predictions: ``raw`` ``q(x) = exp(eta - log Z)``, ``logistic``
    ``H q / (1 + H q)`` and ``cloglog`` ``1 - exp(-H q)`` with ``H = exp(entropy)``
    (Phillips et al. 2017).

    Examples
    --------
    >>> fit = maxent_fit([[1.0], [0.8]], [[0.0], [0.5], [1.0]], add_presence=False)
    >>> [round(v, 6) for v in maxent_predict(fit, [[0.0], [1.0]], output="raw")]
    [0.026297, 0.826297]
    """
    lam = fit["lambdas"]
    out = []
    H = math.exp(fit["entropy"])
    for r in features:
        q = math.exp(ssum(float(a) * b for a, b in zip(r, lam)) - fit["log_normaliser"])
        if output == "raw":
            out.append(q)
        elif output == "logistic":
            out.append(H * q / (1.0 + H * q))
        else:
            out.append(1.0 - math.exp(-H * q))
    return out


def _hull(pts):
    P = sorted(set((float(a), float(b)) for a, b in pts))
    if len(P) <= 2:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _area(poly):
    n = len(poly)
    return abs(ssum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2.0


def _inside(poly, p):
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def _union_area(polys):
    # exact: the union's vertical cross-section length is linear between events (vertices, edge crossings)
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return 0.0
    edges = [(p[i], p[(i + 1) % len(p)]) for p in polys for i in range(len(p))]
    xs = set(v[0] for p in polys for v in p)
    for a in range(len(edges)):
        (x1, y1), (x2, y2) = edges[a]
        for b in range(a + 1, len(edges)):
            (x3, y3), (x4, y4) = edges[b]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                xs.add(x1 + t * (x2 - x1))
    xs = sorted(xs)
    total = 0.0
    for x0, x1 in zip(xs[:-1], xs[1:]):
        if x1 <= x0:
            continue
        xm = (x0 + x1) / 2.0
        ivs = []
        for p in polys:
            ys = []
            for i in range(len(p)):
                (ax, ay), (bx, by) = p[i], p[(i + 1) % len(p)]
                if (ax - xm) * (bx - xm) < 0:
                    ys.append(ay + (by - ay) * (xm - ax) / (bx - ax))
            if len(ys) >= 2:
                ivs.append((min(ys), max(ys)))
        ivs.sort()
        length, cur_lo, cur_hi = 0.0, None, None
        for lo, hi in ivs:
            if cur_hi is None or lo > cur_hi:
                if cur_hi is not None:
                    length += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        if cur_hi is not None:
            length += cur_hi - cur_lo
        total += length * (x1 - x0)
    return total


def locoh_home_range(points, k: int, *, levels=(0.5, 0.95)) -> RichResult:
    r"""k-LoCoH home range (Getz and Wilmers 2004): unions of local nearest-neighbour convex hulls.

    Each point's hull is the convex hull of the point and its ``k - 1``
    nearest neighbours; hulls are sorted by area and merged smallest first;
    the ``q`` isopleth is the union at the first step whose union covers at
    least a fraction ``q`` of the points. Union areas are exact (vertical
    sweep with events at vertices and edge crossings).

    References
    ----------
    Getz, W. M. and Wilmers, C. C. (2004). A local nearest-neighbor
    convex-hull construction of home ranges and utilization distributions.
    Ecography 27, 489-505. Getz, W. M. et al. (2007). LoCoH: nonparameteric
    kernel methods for constructing home ranges and utilization
    distributions. PLoS ONE 2(2), e207.

    Examples
    --------
    >>> pts = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (5, 5)]
    >>> r = locoh_home_range(pts, 4, levels=(0.5,))
    >>> r.areas
    [0.5]
    """
    pts = [(float(a), float(b)) for a, b in points]
    n = len(pts)
    hulls = []
    for i in range(n):
        order = sorted(range(n), key=lambda j: (math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]), j))
        hulls.append(_hull([pts[j] for j in order[:k]]))
    areas = [_area(h) if len(h) >= 3 else 0.0 for h in hulls]
    order = sorted(range(n), key=lambda i: (areas[i], i))
    covered = [False] * n
    out_areas, out_n, cover = [], [], []
    lv = sorted(levels)
    li = 0
    used = []
    for step, h in enumerate(order, start=1):
        used.append(h)
        for p in range(n):
            if not covered[p] and _inside(hulls[h], pts[p]):
                covered[p] = True
        frac = sum(covered) / n
        while li < len(lv) and frac >= lv[li]:
            out_areas.append(_union_area([hulls[q] for q in used]))
            out_n.append(step)
            cover.append(frac)
            li += 1
        if li == len(lv):
            break
    return RichResult(
        payload={
            "levels": lv[: len(out_areas)],
            "areas": out_areas,
            "n_hulls": out_n,
            "coverage": cover,
            "hull_areas": areas,
        }
    )


def cheatsheet() -> str:
    return "maxent_features / maxent_fit / maxent_predict / locoh_home_range -> species distribution and home range."
