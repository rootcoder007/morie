# morie.fn -- function file (rootcoder007/morie)
"""Space-time local statistics: the Getis-Ord Gi* z-score over space-time neighbourhoods (emerging
hot spots), the bivariate Moran's I (global and local, with a permutation test) and polynomial
space-time trend surfaces."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["st_getis_ord", "bivariate_moran", "st_trend_surface"]


def st_getis_ord(z, coords, *, distance: float, time_window: int = 1) -> RichResult:
    r"""Getis-Ord ``Gi*`` z-scores of a space-time cube (``z[t][i]``) with binary space-time neighbourhoods.

    The neighbourhood of location ``i`` at time ``t`` holds every ``(j, s)``
    with ``|s_i - s_j| <= distance`` (``i`` itself included) and ``|t - s| <=
    time_window`` (the space-time "bins" of emerging hot-spot analysis).
    With ``n = N T`` values, mean ``xbar`` and ``S = sqrt(sum x^2 / n -
    xbar^2)``: ``G*_i = (sum_j w_ij x_j - xbar W_i) / (S sqrt((n S1_i -
    W_i^2) / (n - 1)))``, ``W_i = sum_j w_ij``, ``S1_i = sum_j w_ij^2`` (Ord
    and Getis 1995), as ``spdep::localG`` on the space-time neighbours with
    self-inclusion.

    References
    ----------
    Ord, J. K. and Getis, A. (1995). Local spatial autocorrelation
    statistics: distributional issues and an application. *Geographical
    Analysis*, 27(4), 286-306.

    Examples
    --------
    >>> r = st_getis_ord([[1.0, 1.0], [1.0, 5.0]], [(0, 0), (10, 0)], distance=1, time_window=0)
    >>> r.z[1][1] > r.z[0][0]
    True
    """
    Z = [[float(v) for v in row] for row in z]
    P = [tuple(float(v) for v in p) for p in coords]
    T, N = len(Z), len(P)
    x = [v for row in Z for v in row]
    n = len(x)
    xbar = ssum(x) / n
    S = math.sqrt(ssum(v * v for v in x) / n - xbar * xbar)
    nb = [[j for j in range(N) if math.dist(P[i], P[j]) <= distance] for i in range(N)]
    out = []
    for t in range(T):
        row = []
        for i in range(N):
            vals = [Z[s][j] for s in range(max(0, t - time_window), min(T, t + time_window + 1)) for j in nb[i]]
            W = len(vals)
            num = ssum(vals) - xbar * W
            den = S * math.sqrt((n * W - W * W) / (n - 1))
            row.append(num / den)
        out.append(row)
    return RichResult(payload={"z": out, "mean": xbar, "sd": S})


def bivariate_moran(x, y, W, *, nsim: int = 499, seed: int = 1) -> RichResult:
    r"""Bivariate Moran's I ``I_B = sum_i x_i (W y)_i / sum_i x_i^2`` of standardised ``x`` and ``y`` (Wartenberg 1985), as ``spdep::moran_bv``.

    ``x`` and ``y`` are scaled to mean 0 and unit sample variance (``n -
    1``); the local statistics are ``I_i = x_i (W y)_i`` (Anselin, Syabri and
    Smirnov 2002). The p-value permutes ``y`` over locations (Philox
    Fisher-Yates, stream ``s``): ``(1 + #{|I_sim| >= |I|}) / (1 + nsim)``.

    References
    ----------
    Wartenberg, D. (1985). Multivariate spatial correlation: a method for
    exploratory geographical analysis. *Geographical Analysis*, 17(4),
    263-283.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> round(bivariate_moran([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], W, nsim=9).statistic, 12)
    0.4
    """
    xv = [float(v) for v in x]
    yv = [float(v) for v in y]
    Wm = [[float(v) for v in r] for r in W]
    n = len(xv)

    def std(v):
        m = ssum(v) / n
        s = math.sqrt(ssum((a - m) ** 2 for a in v) / (n - 1))
        return [(a - m) / s for a in v]

    xs, ys = std(xv), std(yv)
    sxx = ssum(v * v for v in xs)

    def stat(yy):
        lag = [ssum(a * b for a, b in zip(r, yy)) for r in Wm]
        return ssum(a * b for a, b in zip(xs, lag)) / sxx, [a * b for a, b in zip(xs, lag)]

    moran, local = stat(ys)
    sims = []
    for s in range(nsim):
        u = [float(v) for v in random_uniform(n, seed=seed, stream=s)]
        p = list(ys)
        for t in range(n - 1):
            k = t + int(u[t] * (n - t))
            p[t], p[k] = p[k], p[t]
        sims.append(stat(p)[0])
    return RichResult(
        payload={
            "statistic": moran,
            "local": local,
            "simulated": sims,
            "p_value": (1 + sum(1 for v in sims if abs(v) >= abs(moran))) / (1 + nsim),
        }
    )


def st_trend_surface(z, coords, times, *, degree: int = 2, time_degree: int = 1) -> RichResult:
    r"""Polynomial space-time trend surface ``z = sum b_abc x^a y^b t^c`` (``a + b <= degree``, ``c <= time_degree``) by least squares.

    Coordinates and times are centred (not scaled) before forming the
    monomials; returns the coefficients with the term exponents, fitted
    values, residuals, ``R^2`` and the residual variance (Krumbein 1959;
    Cressie 1993, section 3.4).

    References
    ----------
    Krumbein, W. C. (1959). Trend surface analysis of contour-type maps with
    irregular control-point spacing. *Journal of Geophysical Research*,
    64(7), 823-834.

    Examples
    --------
    >>> r = st_trend_surface([1.0, 2.0, 4.0, 5.0], [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 0, 0], degree=1, time_degree=0)
    >>> round(r.r2, 12)
    1.0
    """
    zv = [float(v) for v in z]
    P = [tuple(float(v) for v in p) for p in coords]
    tv = [float(v) for v in times]
    n = len(zv)
    mx, my, mt = ssum(p[0] for p in P) / n, ssum(p[1] for p in P) / n, ssum(tv) / n
    terms = [
        (a, b, c)
        for c in range(time_degree + 1)
        for tot in range(degree + 1)
        for a in range(tot, -1, -1)
        for b in [tot - a]
    ]
    X = [[(p[0] - mx) ** a * (p[1] - my) ** b * (t - mt) ** c for a, b, c in terms] for p, t in zip(P, tv)]
    k = len(terms)
    XtX = [[ssum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Xtz = [ssum(X[r][i] * zv[r] for r in range(n)) for i in range(k)]
    A = [[float(v) for v in row] for row in inverse(XtX)]
    beta = [ssum(A[i][j] * Xtz[j] for j in range(k)) for i in range(k)]
    fit = [ssum(X[r][i] * beta[i] for i in range(k)) for r in range(n)]
    res = [a - b for a, b in zip(zv, fit)]
    mz = ssum(zv) / n
    rss = ssum(v * v for v in res)
    return RichResult(
        payload={
            "coefficients": beta,
            "terms": terms,
            "fitted": fit,
            "residuals": res,
            "r2": 1 - rss / ssum((v - mz) ** 2 for v in zv),
            "sigma2": rss / (n - k),
        }
    )


def cheatsheet() -> str:
    return "st_getis_ord / bivariate_moran / st_trend_surface -> space-time hot spots, bivariate Moran, trend surfaces."
