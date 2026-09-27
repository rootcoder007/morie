# morie.fn -- function file (rootcoder007/morie)
"""Simple sequential inhibition (random sequential adsorption) in a rectangle."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rng import random_uniform


def sequential_inhibition(r, window, n=None, *, max_failures: int = 1000, seed: int = 0) -> DescriptiveResult:
    """Simple sequential inhibition process of Diggle, Besag and Gleaves (1976).

    Points are proposed uniformly in the window, one at a time, and a
    proposal is kept only if it lies at least ``r`` from every point kept
    so far. The process stops when ``n`` points are kept or, if ``n`` is
    ``None``, after ``max_failures`` consecutive rejections (an
    approximation to the saturated, jammed state; for discs of diameter
    ``r`` the jamming coverage of random sequential adsorption is about
    0.547, Feder 1980). Proposal ``k`` uses Philox uniforms ``2k`` and
    ``2k + 1`` of stream ``2k // 4096`` (blocks of 4096 uniforms).

    :param r: Inhibition distance.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)``.
    :param n: Target number of points (``None`` to saturate).
    :param max_failures: Consecutive rejections allowed.
    :param seed: Philox key.
    :return: DescriptiveResult; ``value`` is the points; ``extra`` has
        ``proposals``, ``n``, ``saturated`` (stopped on failures) and
        ``coverage`` (area fraction of discs of radius ``r / 2``, ignoring
        the parts outside the window).

    References
    ----------
    Diggle, P. J., Besag, J. E. and Gleaves, J. T. (1976). Statistical
    analysis of spatial point patterns by means of distance methods.
    Biometrics 32, 659-667.

    Feder, J. (1980). Random sequential adsorption. Journal of Theoretical
    Biology 87, 237-254.

    Examples
    --------
    >>> r = sequential_inhibition(0.2, (0, 1, 0, 1), n=5, seed=1)
    >>> len(r.value), min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a in r.value for b in r.value if a != b) >= 0.2
    (5, True)
    """
    x0, x1, y0, y1 = (float(v) for v in window)
    rr = float(r) ** 2
    pts, k, fails, block, buf = [], 0, 0, -1, []
    while (n is None or len(pts) < n) and fails < max_failures:
        if 2 * k // 4096 != block:
            block = 2 * k // 4096
            buf = [float(v) for v in random_uniform(4096, seed=seed, stream=block)]
        off = 2 * k - 4096 * block
        u = [x0 + buf[off] * (x1 - x0), y0 + buf[off + 1] * (y1 - y0)]
        k += 1
        if all((u[0] - p[0]) ** 2 + (u[1] - p[1]) ** 2 >= rr for p in pts):
            pts.append(u)
            fails = 0
        else:
            fails += 1
    cov = len(pts) * math.pi * (float(r) / 2) ** 2 / ((x1 - x0) * (y1 - y0))
    return DescriptiveResult(
        name="sequential_inhibition",
        value=pts,
        extra={"proposals": k, "n": len(pts), "saturated": n is None or len(pts) < n, "coverage": cov},
    )


seqinh = sequential_inhibition


def cheatsheet() -> str:
    return "sequential_inhibition(r, window, n) -> simple sequential inhibition (RSA) point pattern"
