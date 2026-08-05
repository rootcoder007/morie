# morie.fn -- function file (rootcoder007/morie)
"""Pair correlation function g(r) for point patterns."""

import math

from . import _schab_pp as pp

from ._richresult import RichResult

__all__ = ["pair_correlation_function"]


def pair_correlation_function(points, window, r, h=None):
    """Pair correlation function from the derivative of Ripley's K.

    Formula: ``g(r) = K'(r) / (2 pi r)``.  ``K`` is estimated with the
    reduced-sample (border) correction and differentiated by a central
    difference of half-width ``h``,

        g_hat(r) = (K_hat(r + h) - K_hat(r - h)) / (2 h) / (2 pi r),

    which is deterministic -- no kernel smoothing, no bandwidth rule
    that depends on the data.  Under complete spatial randomness
    ``K(r) = pi r^2`` so ``g(r) = 1`` at every radius; values above one
    mean clustering at that scale and below one regularity.

    Parameters
    ----------
    points : array-like, shape (n, 2)
        Point coordinates.
    window : sequence or None
        ``(xmin, ymin, xmax, ymax)`` or an (m, 2) vertex array;
        ``None`` takes the bounding box of ``points``.
    r : array-like
        Radii, strictly positive (``g`` has a ``1/r`` factor).
    h : float or None
        Half-width of the central difference.  Defaults to one quarter
        of the smallest radius supplied, which keeps ``r - h`` positive.

    Returns
    -------
    RichResult
        ``g`` (values on the grid), ``r``, ``K``, ``h``,
        ``estimate`` (``g`` at the first radius), ``lambda_hat``, ``n``.

    References
    ----------
    Stoyan, D. & Stoyan, H. (1994).  Fractals, Random Shapes and Point
    Fields, Wiley, chapter 14; Diggle, P. J. (2003).  Statistical
    Analysis of Spatial Point Patterns, 2nd edition, Arnold,
    section 4.3.
    """
    p = pp.as_points(points)
    n = len(p)
    if n < 2:
        raise ValueError("pair_correlation_function: need at least two points")
    region = pp.as_region(window, p)
    rs = [float(v) for v in (r if hasattr(r, "__len__") else [r])]
    if not rs:
        raise ValueError("pair_correlation_function: r is empty")
    if any(v <= 0.0 for v in rs):
        raise ValueError("pair_correlation_function: r must be strictly positive")
    hh = float(h) if h is not None else min(rs) / 4.0
    if hh <= 0.0:
        raise ValueError("pair_correlation_function: h must be positive")
    lo = [v - hh for v in rs]
    hi = [v + hh for v in rs]
    Klo = [float(v) for v in pp.k_function(p, region, lo, correction="border")]
    Khi = [float(v) for v in pp.k_function(p, region, hi, correction="border")]
    Kat = [float(v) for v in pp.k_function(p, region, rs, correction="border")]
    g = [(Khi[i] - Klo[i]) / (2.0 * hh) / (2.0 * math.pi * rs[i])
         for i in range(len(rs))]
    return RichResult(payload={
        "g": g, "r": rs, "K": Kat, "h": hh, "estimate": g[0],
        "lambda_hat": pp.intensity(p, region), "n": n,
        "method": "Pair correlation function from K'(r) / (2 pi r)"})


def cheatsheet():
    return "pcfunc: Pair correlation function g(r) = K'(r) / (2 pi r)"


paircorrelationfunction = pair_correlation_function
