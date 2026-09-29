# morie.fn -- function file (rootcoder007/morie)
"""Spatial kernel density estimation: bivariate Gaussian kernel density on a grid."""

from __future__ import annotations

import math

from ._richresult import RichResult


def _q7(s, p):
    """Type-7 quantile of sorted ``s`` (R's default)."""
    idx = (len(s) - 1) * p
    lo = math.floor(idx)
    h = idx - lo
    return s[lo] if h == 0 else (1 - h) * s[lo] + h * s[lo + 1]


def _sd(v):
    m = math.fsum(v) / len(v)
    return math.sqrt(math.fsum((t - m) ** 2 for t in v) / (len(v) - 1))


def _bw(v, method):
    n = len(v)
    if method == "scott":
        return _sd(v) * n ** (-1.0 / 6.0)
    s = sorted(v)
    iqr = (_q7(s, 0.75) - _q7(s, 0.25)) / 1.34
    return 1.06 * min(_sd(v), iqr) * n ** (-0.2)


def _grid(a, b, m):
    if m == 1:
        return [a]
    by = (b - a) / (m - 1)
    g = [a + i * by for i in range(m)]
    g[-1] = b
    return g


def spatial_kde(data, *, method: str = "nrd", h=None, n=25, lims=None) -> RichResult:
    r"""Bivariate Gaussian kernel density estimate of a point pattern, evaluated on a regular grid.

    ``f(u, v) = (1 / (N h_x h_y)) sum_k phi((u - x_k) / h_x) phi((v - y_k) / h_y)``
    with ``phi`` the standard normal density and ``h_x, h_y`` the kernel
    standard deviations. ``method="nrd"`` (default) is the normal-reference
    plug-in rule ``h = 1.06 min(s, IQR / 1.34) N^{-1/5}`` per axis (Silverman
    1986, eq. 3.31; ``MASS::bandwidth.nrd / 4``, so that the result equals
    ``MASS::kde2d``); ``method="scott"`` is Scott's rule ``h = s N^{-1/(d+4)}
    = s N^{-1/6}`` for ``d = 2``. ``h`` (scalar or pair of kernel standard
    deviations) overrides the rule. The grid has ``n`` points per axis (scalar
    or pair) over ``lims = (xmin, xmax, ymin, ymax)`` (default the data range).

    Parameters
    ----------
    data : sequence of ``(x, y)`` points.
    method : ``"nrd"`` or ``"scott"`` (``"default"`` means ``"nrd"``).
    h : kernel standard deviations (optional).
    n : grid size per axis.
    lims : grid limits (optional).

    Returns
    -------
    RichResult
        ``x``, ``y`` (grid coordinates), ``z`` (``z[i][j]`` is the density at
        ``(x[i], y[j])``), ``bandwidth`` (``h_x, h_y``) and ``n``.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman and Hall, sections 3.4 and 4.2.

    Scott, D. W. (1992). *Multivariate Density Estimation*. Wiley, section 6.3.

    Venables, W. N. and Ripley, B. D. (2002). *Modern Applied Statistics with
    S*, 4th edn. Springer, section 5.6 (``kde2d``).

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.9), (0.5, 0.4), (0.9, 0.7), (0.3, 0.6)]
    >>> r = spatial_kde(pts, n=3)
    >>> [round(v, 10) for v in r.bandwidth]
    [0.1146666334, 0.17199995]
    >>> round(r.z[1][1], 10)
    1.5835738263
    """
    pts = [(float(p[0]), float(p[1])) for p in data]
    N = len(pts)
    if N < 2:
        raise ValueError("need at least 2 points")
    if method == "default":
        method = "nrd"
    if method not in ("nrd", "scott"):
        raise ValueError("method must be 'nrd' or 'scott'")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if h is None:
        hx, hy = _bw(xs, method), _bw(ys, method)
    else:
        hh = [float(v) for v in h] if isinstance(h, (list, tuple)) else [float(h)] * 2
        hx, hy = hh[0], hh[-1]
    if not (hx > 0 and hy > 0):
        raise ValueError("bandwidths must be positive")
    nn = [int(v) for v in n] if isinstance(n, (list, tuple)) else [int(n)] * 2
    lm = [min(xs), max(xs), min(ys), max(ys)] if lims is None else [float(v) for v in lims]
    gx, gy = _grid(lm[0], lm[1], nn[0]), _grid(lm[2], lm[3], nn[-1])
    c = 1.0 / math.sqrt(2.0 * math.pi)
    ax = [[c * math.exp(-0.5 * ((g - x) / hx) ** 2) for x in xs] for g in gx]
    ay = [[c * math.exp(-0.5 * ((g - y) / hy) ** 2) for y in ys] for g in gy]
    den = N * hx * hy
    z = []
    for a in ax:
        row = []
        for b in ay:
            s = 0.0
            for k in range(N):
                s += a[k] * b[k]
            row.append(s / den)
        z.append(row)
    return RichResult(payload={"x": gx, "y": gy, "z": z, "bandwidth": [hx, hy], "n": N})


spat = spatial_kde


def cheatsheet() -> str:
    return "spatial_kde(points, method='nrd'|'scott', h, n, lims) -> 2-D Gaussian KDE on a grid (MASS::kde2d)."


# compact alias per ledger/NAMING.md
spatialkde = spatial_kde
