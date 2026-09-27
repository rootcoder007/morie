# morie.fn -- function file (rootcoder007/morie)
"""Berman-Diggle cross-validated bandwidth for kernel intensity estimation."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from .ripk import _rect, isotropic_weight

__all__ = ["bandwidth_diggle"]


def bandwidth_diggle(points, window, nr: int = 512, hmax: float | None = None) -> RichResult:
    r"""Gaussian kernel bandwidth by Berman-Diggle cross-validation.

    Diggle (1985) and Berman and Diggle (1989) choose the standard
    deviation ``sigma`` of an isotropic Gaussian kernel intensity estimate
    by minimising an estimate of its mean squared error, which depends on
    the data only through Ripley's K-function:

    .. math::

        M(r) = \frac{1/\lambda - 2 K(r)}{\pi r^2} + \frac{J(r)}{(\pi r^2)^2},
        \qquad J(r) = \int_0^{2r} \phi(t, r)\, dK(t),

    ``phi(t, r) = 2 r^2 (acos(y) - y sqrt(1 - y^2))``, ``y = t/(2r)``,
    with ``sigma = r/2``.  As ``spatstat.explore::bw.diggle`` (rectangular
    window): K is the isotropic (Ripley) estimate with
    ``lambda^2 = n(n-1)/|W|^2`` on ``nr`` equally spaced radii up to
    ``rmax = min(shortside/4, sqrt(1000/(pi lambda)))`` (or ``4 hmax``),
    J is the left-point Stieltjes sum over that grid, M is evaluated for
    ``r <= rmax/2`` and the bandwidth is the grid value minimising it.

    :param points: (n, 2) coordinates.
    :param window: ``(xmin, xmax, ymin, ymax)``.
    :param nr: Number of radii in the K grid.
    :param hmax: Largest bandwidth considered (default from the rule above).
    :return: :class:`RichResult` with ``sigma`` (the bandwidth), ``h``
        (candidate bandwidths), ``criterion`` (``M`` at each), ``J`` and
        ``lambda_hat`` and ``at_boundary`` (the minimum is at an end of the
        grid, where ``bw.diggle`` warns; widen with ``hmax``).

    References
    ----------
    Berman, M. and Diggle, P. (1989). Estimating weighted integrals of the
    second-order intensity of a spatial point process. *Journal of the
    Royal Statistical Society B*, 51(1), 81-92.
    Diggle, P. J. (1985). A kernel method for smoothing point process data.
    *Applied Statistics*, 34(2), 138-147.

    Examples
    --------
    >>> pts = [(0.1, 0.2), (0.4, 0.8), (0.35, 0.3), (0.8, 0.6), (0.7, 0.15), (0.55, 0.5),
    ...        (0.2, 0.65), (0.9, 0.9), (0.15, 0.95), (0.6, 0.85)]
    >>> r = bandwidth_diggle(pts, (0, 1, 0, 1))
    >>> round(r.sigma, 6), r.at_boundary
    (0.062378, True)
    """
    P = [(float(a), float(b)) for a, b in np.asarray(points, dtype=float).tolist()]
    n = len(P)
    if n < 2:
        raise ValueError("need at least two points")
    x0, x1, y0, y1 = _rect(window)
    area = (x1 - x0) * (y1 - y0)
    lam = n / area
    if hmax is not None:
        rmax = 4.0 * float(hmax)
    else:
        rmax = min(math.sqrt(1000.0 / (math.pi * lam)), min(x1 - x0, y1 - y0) / 4.0)
    nr = int(nr)
    r = [rmax * k / (nr - 1) for k in range(nr)]
    # pair distances and 1/w isotropic weights once, then cumulate over the grid
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                d = math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1])
                if d <= rmax:
                    pairs.append((d, 1.0 / isotropic_weight(P[i][0], P[i][1], d, x0, x1, y0, y1)))
    pairs.sort()
    K = []
    acc, p = 0.0, 0
    scale = area / (n * (n - 1.0))
    for h in r:
        while p < len(pairs) and pairs[p][0] <= h:
            acc += pairs[p][1]
            p += 1
        K.append(scale * acc)
    dK = [K[k + 1] - K[k] for k in range(nr - 1)]
    half = r[-1] / 2.0
    ok = [k for k in range(nr) if r[k] <= half]
    J = [0.0] * len(ok)
    for i in ok[1:]:
        tw = 2.0 * r[i]
        s = 0.0
        for j in range(nr - 1):
            y = r[j] / tw
            if y >= 1.0:
                break
            s += (math.acos(y) - y * math.sqrt(1.0 - y * y)) * dK[j]
        J[i] = 2.0 * r[i] * r[i] * s
    crit, hs = [], []
    for i in ok:
        pr2 = math.pi * r[i] * r[i]
        crit.append((1.0 / lam - 2.0 * K[i]) / pr2 + J[i] / pr2**2 if pr2 > 0.0 else float("nan"))
        hs.append(r[i] / 2.0)
    best = min((k for k in range(len(crit)) if crit[k] == crit[k]), key=lambda k: crit[k])
    return RichResult(
        payload={
            "sigma": hs[best],
            "h": hs,
            "criterion": crit,
            "J": J,
            "lambda_hat": lam,
            "at_boundary": best in (1, len(crit) - 1),
        }
    )


def cheatsheet() -> str:
    return "bandwidth_diggle(points, window) -> Berman-Diggle CV bandwidth (spatstat bw.diggle)."
