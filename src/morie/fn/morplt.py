# morie.fn -- function file (rootcoder007/morie)
"""Moran scatterplot data with regression influence diagnostics (Anselin 1996)."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

__all__ = ["moran_scatter"]


def moran_scatter(x, W, y=None) -> RichResult:
    r"""Coordinates and influence measures of the Moran scatterplot.

    The scatterplot pairs ``x`` with its spatial lag ``Wx`` (or with the
    lag ``Wy`` of a second variable); its least-squares slope is Moran's I
    when ``W`` is row-standardised (Anselin 1996).  As ``spdep::moran.plot``,
    the line ``Wx ~ x`` is fitted and each point gets the influence
    measures of ``stats::influence.measures``: ``dfb_1`` and ``dfb_x``
    (DFBETAS), ``dffit``, ``cov_r``, ``cook_d`` and ``hat``, and is flagged
    ``is_inf`` when any of ``|DFBETAS| > 1``, ``|DFFIT| > 3 sqrt(2/(n-2))``,
    ``|1 - COVRATIO| > 6/(n-2)``, ``F(2, n-2)`` cdf of Cook's distance
    ``> 0.5`` or ``hat > 6/n`` holds.

    :param x: Values (n,).
    :param W: Spatial weights (n, n).
    :param y: Optional second variable whose lag is the vertical axis.
    :return: :class:`RichResult` with ``x``, ``wx``, ``slope``,
        ``intercept``, ``dfb_1``, ``dfb_x``, ``dffit``, ``cov_r``,
        ``cook_d``, ``hat``, ``is_inf``.

    References
    ----------
    Anselin, L. (1996). The Moran scatterplot as an ESDA tool to assess
    local instability in spatial association. In Fischer, M., Scholten,
    H. J. and Unwin, D. (eds), *Spatial Analytical Perspectives on GIS*,
    111-125. Taylor and Francis, London.
    Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). *Regression
    Diagnostics*. Wiley, New York.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> round(moran_scatter([1.0, 2.0, 4.0, 3.0], W).slope, 6)
    0.3
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    n = len(xs)
    Wl = np.asarray(W, dtype=float).tolist()
    if len(Wl) != n or any(len(r) != n for r in Wl):
        raise ValueError("W must be n x n with n = len(x)")
    if n < 4:
        raise ValueError("need at least four units")
    src = xs if y is None else [float(v) for v in np.asarray(y, dtype=float).tolist()]
    if len(src) != n:
        raise ValueError("y must have length n")
    wx = [sum(Wl[i][j] * src[j] for j in range(n) if j != i) for i in range(n)]
    mx = sum(xs) / n
    my = sum(wx) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx == 0.0:
        raise ValueError("x is constant")
    b = sum((xs[i] - mx) * (wx[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    e = [wx[i] - a - b * xs[i] for i in range(n)]
    p = 2.0
    rss = sum(v * v for v in e)
    s = (rss / (n - p)) ** 0.5
    # (X'X)^{-1} for columns (1, x)
    sx, sxx2 = sum(xs), sum(v * v for v in xs)
    det = n * sxx2 - sx * sx
    c11, c12, c22 = sxx2 / det, -sx / det, n / det
    out = {k: [] for k in ("dfb_1", "dfb_x", "dffit", "cov_r", "cook_d", "hat", "is_inf")}
    for i in range(n):
        h = 1.0 / n + (xs[i] - mx) ** 2 / sxx
        si = ((rss - e[i] ** 2 / (1.0 - h)) / (n - p - 1.0)) ** 0.5
        g1 = (c11 + c12 * xs[i]) * e[i] / (1.0 - h)
        g2 = (c12 + c22 * xs[i]) * e[i] / (1.0 - h)
        d1 = g1 / (si * c11**0.5)
        d2 = g2 / (si * c22**0.5)
        dff = e[i] * h**0.5 / (si * (1.0 - h))
        cvr = (si / s) ** (2.0 * p) / (1.0 - h)
        cook = (e[i] / (s * (1.0 - h))) ** 2 * h / p
        flag = (
            abs(d1) > 1.0
            or abs(d2) > 1.0
            or abs(dff) > 3.0 * (p / (n - p)) ** 0.5
            or abs(1.0 - cvr) > 3.0 * p / (n - p)
            or float(stats.f.cdf(cook, p, n - p)) > 0.5
            or h > 3.0 * p / n
        )
        for k, v in zip(out, (d1, d2, dff, cvr, cook, h, bool(flag))):
            out[k].append(v)
    return RichResult(payload={"x": xs, "wx": wx, "slope": b, "intercept": a, **out})


def cheatsheet() -> str:
    return "moran_scatter(x, W) -> Moran scatterplot points, slope and influence flags (spdep::moran.plot)."
