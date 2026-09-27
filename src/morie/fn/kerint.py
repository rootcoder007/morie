# morie.fn -- function file (rootcoder007/morie)
"""Gaussian kernel intensity of a point pattern with edge corrections."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import ssum


def _cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def kernel_intensity(
    points, window, sigma, *, at=None, correction: str = "uniform", leaveoneout: bool = True, weights=None
) -> DescriptiveResult:
    """Kernel estimate of the intensity of a point pattern in a rectangle.

    With the isotropic Gaussian kernel ``k(d) = exp(-d^2 / (2 sigma^2)) / (2 pi sigma^2)``
    and the edge mass ``e(u) = [Phi((x1 - u_x)/sigma) - Phi((x0 - u_x)/sigma)]
    [Phi((y1 - u_y)/sigma) - Phi((y0 - u_y)/sigma)]`` (the share of the
    kernel centred at ``u`` that falls inside the window)::

        none     lambda(u) = sum_j w_j k(u - x_j)
        uniform  lambda(u) = sum_j w_j k(u - x_j) / e(u)          (Diggle 1985)
        diggle   lambda(u) = sum_j w_j k(u - x_j) / e(x_j)        (Jones 1993)

    Evaluated at the data points (``at=None``, leaving each point out of
    its own sum by default) or at given locations. These are
    ``spatstat.explore::density.ppp`` with ``at = "points"`` and its
    ``edge`` and ``diggle`` options (which also drop kernel mass beyond
    ``8 sigma``, a relative change below ``1e-13``).

    :param points: (n, 2) event locations inside ``window``.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)``.
    :param sigma: Kernel standard deviation.
    :param at: Optional (m, 2) evaluation locations.
    :param correction: ``"none"``, ``"uniform"`` or ``"diggle"``.
    :param leaveoneout: At the data points, omit each point's own kernel.
    :param weights: Optional point weights (default 1).
    :return: DescriptiveResult; ``value`` is the intensity at each location;
        ``extra`` has ``edge`` (``e`` at the evaluation locations),
        ``correction``, ``sigma``.

    References
    ----------
    Diggle, P. J. (1985). A kernel method for smoothing point process data.
    Applied Statistics 34, 138-147.

    Jones, M. C. (1993). Simple boundary correction for kernel density
    estimation. Statistics and Computing 3, 135-146.

    Examples
    --------
    >>> r = kernel_intensity([[0.2, 0.3], [0.4, 0.5], [0.7, 0.2]], (0, 1, 0, 1), 0.2)
    >>> [round(v, 9) for v in r.value]
    [2.061408548, 1.953885829, 0.730843918]
    """
    P = [[float(v) for v in p] for p in points]
    x0, x1, y0, y1 = (float(v) for v in window)
    sig = float(sigma)
    w = [1.0] * len(P) if weights is None else [float(v) for v in weights]
    if correction not in ("none", "uniform", "diggle"):
        raise ValueError("correction must be 'none', 'uniform' or 'diggle'")

    def edge(u):
        return (_cdf((x1 - u[0]) / sig) - _cdf((x0 - u[0]) / sig)) * (_cdf((y1 - u[1]) / sig) - _cdf((y0 - u[1]) / sig))

    c = 1.0 / (2.0 * math.pi * sig * sig)
    if correction == "diggle":
        w = [a / edge(p) for a, p in zip(w, P)]
    locs = P if at is None else [[float(v) for v in u] for u in at]
    own = at is None and leaveoneout
    out, eu = [], []
    for k, u in enumerate(locs):
        s = ssum(
            w[j] * c * math.exp(-((u[0] - p[0]) ** 2 + (u[1] - p[1]) ** 2) / (2.0 * sig * sig))
            for j, p in enumerate(P)
            if not (own and j == k)
        )
        e = edge(u)
        eu.append(e)
        out.append(s / e if correction == "uniform" else s)
    return DescriptiveResult(
        name="kernel_intensity", value=out, extra={"edge": eu, "correction": correction, "sigma": sig}
    )


kerint = kernel_intensity


def cheatsheet() -> str:
    return "kernel_intensity(points, window, sigma) -> Gaussian kernel intensity with uniform or Jones-Diggle edge correction"
