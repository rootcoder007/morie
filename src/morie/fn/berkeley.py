# morie.fn -- function file (rootcoder007/morie)
"""Berkeley Earth averaging: a regional mean temperature series, station baselines and the weather-field
correlation model estimated jointly by iterated ordinary block kriging and weighted least squares."""

from __future__ import annotations

import math

from ._qpcore import solve
from ._richresult import RichResult

__all__ = ["berkeley_earth"]


def _lsum(v):
    s = 0.0
    for a in v:
        s += a
    return s


def _dist(a, b, spherical):
    if not spherical:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    la1, lo1, la2, lo2 = (math.radians(v) for v in (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _corr(d, c, L):
    return 1.0 if d == 0 else c * math.exp(-d / L)


def _fit_corr(pairs):
    """Weighted least squares of r = c exp(-d / L) over station pairs (d, r, n): grid, then golden section on L."""
    ds = [d for d, _, _ in pairs if d > 0]
    lo, hi = min(ds) / 10.0, max(ds) * 10.0

    def sse_c(L):
        num = den = 0.0
        for d, r, n in pairs:
            e = math.exp(-d / L)
            num += n * r * e
            den += n * e * e
        c = min(max(num / den, 0.0), 1.0) if den > 0 else 0.0
        s = 0.0
        for d, r, n in pairs:
            v = r - c * math.exp(-d / L)
            s += n * v * v
        return s, c

    K = 60
    grid = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * k / (K - 1)) for k in range(K)]
    vals = [sse_c(L)[0] for L in grid]
    kb = min(range(K), key=lambda k: (vals[k], k))
    a, b = grid[max(kb - 1, 0)], grid[min(kb + 1, K - 1)]
    g = (math.sqrt(5.0) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = sse_c(x1)[0], sse_c(x2)[0]
    for _ in range(80):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = sse_c(x1)[0]
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = sse_c(x2)[0]
    L = (a + b) / 2
    s, c = sse_c(L)
    return c, L, s


def _pair_corr(resid, D):
    pairs = []
    m = len(resid)
    for i in range(m):
        for j in range(i + 1, m):
            idx = [t for t in range(len(resid[i])) if resid[i][t] is not None and resid[j][t] is not None]
            if len(idx) < 3:
                continue
            xa = [resid[i][t] for t in idx]
            xb = [resid[j][t] for t in idx]
            ma = _lsum(xa) / len(xa)
            mb = _lsum(xb) / len(xb)
            sab = saa = sbb = 0.0
            for u, v in zip(xa, xb):
                sab += (u - ma) * (v - mb)
                saa += (u - ma) ** 2
                sbb += (v - mb) ** 2
            if saa > 0 and sbb > 0:
                pairs.append((D[i][j], sab / math.sqrt(saa * sbb), float(len(idx))))
    return pairs


def berkeley_earth(
    stations, *, series=None, grid=None, spherical: bool = False, n_iter: int = 20, tol: float = 1e-10
) -> RichResult:
    r"""Berkeley Earth regional average ``T_i(t) = theta(t) + b_i + W(x_i, t)`` (Rohde et al. 2013).

    ``stations`` are coordinates ((x, y), or (lat, lon) with
    ``spherical=True`` and great-circle kilometres) and ``series[i][t]``
    the station temperatures (``None`` or NaN when missing). Starting from
    station means as baselines, each iteration (1) fits the weather-field
    correlation ``R(d) = c exp(-d / L)`` (nugget ``1 - c``) by weighted
    least squares to the pairwise correlations of the current residuals
    ``W = T - theta - b``, weights the overlap lengths; (2) for every time
    estimates ``theta(t) = sum_i w_i (T_i(t) - b_i)`` by ordinary block
    kriging of the domain mean over ``grid`` (default: a 10 x 10 grid on
    the stations' bounding box): ``[R 1; 1' 0][w; mu] = [rbar; 1]`` with
    ``rbar_i`` the mean correlation between station ``i`` and the grid;
    (3) re-estimates ``b_i`` as the station mean of ``T_i - theta`` and
    centres ``theta`` to mean zero. The block-kriging standard error is
    ``sigma_W sqrt(Rbar_gg - w' rbar - mu)``.

    References
    ----------
    Rohde, R. et al. (2013). Berkeley Earth temperature averaging process.
    *Geoinformatics and Geostatistics: An Overview*, 1(2).
    Rohde, R. et al. (2013). A new estimate of the average Earth surface
    land temperature spanning 1753 to 2011. *Geoinformatics and
    Geostatistics: An Overview*, 1(1).

    Examples
    --------
    >>> st = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    >>> th = [0.0, 0.3, -0.2, 0.5, 0.1, 0.4]
    >>> w = [[0.1, -0.1, 0.0, 0.05, -0.05, 0.0], [0.0, 0.1, -0.1, 0.0, 0.05, -0.05],
    ...      [-0.05, 0.0, 0.1, -0.1, 0.0, 0.05], [0.05, -0.05, 0.0, 0.1, -0.1, 0.0],
    ...      [0.0, 0.05, -0.05, 0.0, 0.1, -0.1]]
    >>> T = [[10 + 2 * i + th[t] + w[i][t] for t in range(6)] for i in range(5)]
    >>> r = berkeley_earth(st, series=T)
    >>> [round(v, 4) for v in r.theta]
    [-0.1633, 0.1167, -0.3933, 0.3267, -0.0833, 0.1967]
    """
    X = [tuple(float(v) for v in s) for s in stations]
    if series is None:
        raise ValueError("series is required")
    S = [[None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in row] for row in series]
    m, nt = len(X), len(S[0])
    if len(S) != m or any(len(r) != nt for r in S):
        raise ValueError("series must have one row per station, all of the same length")
    if grid is None:
        xs = [p[0] for p in X]
        ys = [p[1] for p in X]
        grid = [
            (min(xs) + (max(xs) - min(xs)) * (a + 0.5) / 10, min(ys) + (max(ys) - min(ys)) * (b + 0.5) / 10)
            for a in range(10)
            for b in range(10)
        ]
    G = [tuple(float(v) for v in g) for g in grid]
    D = [[_dist(X[i], X[j], spherical) for j in range(m)] for i in range(m)]
    DG = [[_dist(X[i], g, spherical) for g in G] for i in range(m)]
    DGG = [[_dist(g, h, spherical) for h in G] for g in G]
    b = []
    for row in S:
        v = [x for x in row if x is not None]
        if not v:
            raise ValueError("every station needs at least one value")
        b.append(_lsum(v) / len(v))
    theta = [0.0] * nt
    se = [0.0] * nt
    c, L, sse = 1.0, 1.0, 0.0
    it = 0
    for it in range(1, int(n_iter) + 1):
        resid = [[None if S[i][t] is None else S[i][t] - theta[t] - b[i] for t in range(nt)] for i in range(m)]
        pairs = _pair_corr(resid, D)
        if not pairs:
            raise ValueError("no station pair overlaps in three or more times")
        c, L, sse = _fit_corr(pairs)
        allr = [v for row in resid for v in row if v is not None]
        mr = _lsum(allr) / len(allr)
        s2 = _lsum((v - mr) ** 2 for v in allr) / max(len(allr) - 1, 1)
        rgg = 0.0
        for row in DGG:
            for d in row:
                rgg += _corr(d, c, L)
        rgg /= len(G) ** 2
        new = []
        for t in range(nt):
            av = [i for i in range(m) if S[i][t] is not None]
            if not av:
                new.append(math.nan)
                se[t] = math.nan
                continue
            k = len(av)
            A = [[_corr(D[i][j], c, L) for j in av] + [1.0] for i in av] + [[1.0] * k + [0.0]]
            rbar = []
            for i in av:
                s = 0.0
                for d in DG[i]:
                    s += _corr(d, c, L)
                rbar.append(s / len(G))
            sol = solve(A, rbar + [1.0])
            w, mu = sol[:k], sol[k]
            th = 0.0
            wr = 0.0
            for q, i in enumerate(av):
                th += w[q] * (S[i][t] - b[i])
                wr += w[q] * rbar[q]
            new.append(th)
            se[t] = math.sqrt(max(s2 * (rgg - wr - mu), 0.0))
        ok = [v for v in new if not math.isnan(v)]
        shift = _lsum(ok) / len(ok)
        new = [v - shift for v in new]
        nb = []
        for i in range(m):
            v = [S[i][t] - new[t] for t in range(nt) if S[i][t] is not None and not math.isnan(new[t])]
            nb.append(_lsum(v) / len(v) if v else b[i])
        change = max(
            max(abs(a - q) for a, q in zip(new, theta) if not math.isnan(a)),
            max(abs(a - q) for a, q in zip(nb, b)),
        )
        theta, b = new, nb
        if change < tol:
            break
    return RichResult(
        payload={
            "theta": theta,
            "theta_se": se,
            "baselines": b,
            "correlation_c": c,
            "correlation_length": L,
            "nugget": 1.0 - c,
            "correlation_sse": sse,
            "iterations": it,
        }
    )


def cheatsheet() -> str:
    return "berkeley_earth(stations, series=T) -> regional mean theta(t), baselines, correlation model (Rohde 2013)."
