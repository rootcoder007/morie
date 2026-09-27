# morie.fn -- function file (rootcoder007/morie)
"""Log-linear Poisson point process: Berman-Turner fit and simulation."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import solve, ssum
from ._rng import random_uniform


def _terms(x, y, degree):
    z = [1.0]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            z.append(x ** (d - k) * y**k)
    return z


def poisson_process_fit(
    points,
    window,
    *,
    degree: int = 1,
    nx: int = 12,
    ny: int = 12,
    simulate: int = 0,
    seed: int = 0,
    max_iter: int = 100,
) -> DescriptiveResult:
    """Maximum likelihood for a Poisson process with log-polynomial intensity.

    ``log lambda(u) = theta' z(u)`` with ``z`` the monomials in the
    coordinates up to ``degree`` (``degree = 0`` is the homogeneous
    process, ``lambda = n / |W|``). The likelihood
    ``sum_i log lambda(x_i) - int_W lambda`` is approximated by the
    Berman-Turner quadrature: the data points plus the ``nx`` by ``ny``
    grid of tile centres, each quadrature point weighted by its tile's area
    shared among the quadrature points in that tile; the approximation is
    a weighted Poisson regression, solved by Newton's method. With
    ``simulate > 0``, that many patterns are drawn from the fitted process
    by Lewis-Shedler thinning of a homogeneous process at the maximum of
    the fitted intensity over the window's corners and quadrature points
    (Philox; the Poisson count is inverted in independent parts of mean at
    most 500 so that ``exp(-mean)`` stays representable).

    :param points: (n, 2) event locations inside ``window``.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)``.
    :param degree: Polynomial degree of the log intensity.
    :param nx: Dummy grid columns.
    :param ny: Dummy grid rows.
    :param simulate: Number of patterns to simulate from the fit.
    :param seed: Philox key for the simulation.
    :param max_iter: Newton iterations.
    :return: DescriptiveResult; ``value`` is ``theta``; ``extra`` has ``se``
        (from the inverse information), ``loglik`` (the quadrature
        log-likelihood), ``expected_count`` (``int_W lambda`` by quadrature)
        and ``simulated`` patterns.

    References
    ----------
    Berman, M. and Turner, T. R. (1992). Approximating point process
    likelihoods with GLIM. Applied Statistics 41, 31-38.

    Lewis, P. A. W. and Shedler, G. S. (1979). Simulation of
    nonhomogeneous Poisson processes by thinning. Naval Research Logistics
    Quarterly 26, 403-413.

    Examples
    --------
    >>> r = poisson_process_fit([[0.1, 0.2], [0.5, 0.5], [0.9, 0.7]], (0, 1, 0, 2), degree=0)
    >>> round(math.exp(r.value[0]), 12)
    1.5
    """
    P = [[float(v) for v in p] for p in points]
    x0, x1, y0, y1 = (float(v) for v in window)
    n = len(P)
    dummy = [[x0 + (a + 0.5) * (x1 - x0) / nx, y0 + (b + 0.5) * (y1 - y0) / ny] for b in range(ny) for a in range(nx)]
    quad = P + dummy
    ind = [1.0] * n + [0.0] * len(dummy)

    def tile(p):
        a = min(int((p[0] - x0) / (x1 - x0) * nx), nx - 1)
        b = min(int((p[1] - y0) / (y1 - y0) * ny), ny - 1)
        return a, b

    tiles = [tile(p) for p in quad]
    counts = {}
    for t in tiles:
        counts[t] = counts.get(t, 0) + 1
    tarea = (x1 - x0) * (y1 - y0) / (nx * ny)
    w = [tarea / counts[t] for t in tiles]
    Z = [_terms(p[0], p[1], degree) for p in quad]
    k = len(Z[0])
    theta = [math.log(max(n, 1) / ((x1 - x0) * (y1 - y0)))] + [0.0] * (k - 1)
    for _ in range(max_iter):
        lam = [math.exp(ssum(a * b for a, b in zip(theta, z))) for z in Z]
        g = [ssum(Z[i][r] * (ind[i] - w[i] * lam[i]) for i in range(len(quad))) for r in range(k)]
        H = [[ssum(w[i] * lam[i] * Z[i][r] * Z[i][c] for i in range(len(quad))) for c in range(k)] for r in range(k)]
        step = solve(H, g)
        theta = [a + b for a, b in zip(theta, step)]
        if max(abs(t) for t in step) < 1e-12:
            break
    lam = [math.exp(ssum(a * b for a, b in zip(theta, z))) for z in Z]
    H = [[ssum(w[i] * lam[i] * Z[i][r] * Z[i][c] for i in range(len(quad))) for c in range(k)] for r in range(k)]
    se = [math.sqrt(solve(H, [1.0 if c == r else 0.0 for c in range(k)])[r]) for r in range(k)]
    loglik = ssum(math.log(lam[i]) for i in range(n)) - ssum(w[i] * lam[i] for i in range(len(quad)))
    sims = []
    if simulate > 0:
        corners = [[x0, y0], [x0, y1], [x1, y0], [x1, y1]]
        lmax = max(math.exp(ssum(a * b for a, b in zip(theta, _terms(p[0], p[1], degree)))) for p in corners + quad)
        mean = lmax * (x1 - x0) * (y1 - y0)
        parts = max(1, math.ceil(mean / 500.0))
        for s in range(simulate):
            cnt = 0
            for u in (float(v) for v in random_uniform(parts, seed=seed, stream=2 * s)):
                k, prob = 0, math.exp(-mean / parts)
                cum = prob
                while u > cum and prob > 0:
                    k += 1
                    prob *= mean / parts / k
                    cum += prob
                cnt += k
            v = [float(t) for t in random_uniform(3 * cnt, seed=seed, stream=2 * s + 1)] if cnt else []
            pat = []
            for j in range(cnt):
                px = x0 + v[3 * j] * (x1 - x0)
                py = y0 + v[3 * j + 1] * (y1 - y0)
                lam_p = math.exp(ssum(a * b for a, b in zip(theta, _terms(px, py, degree))))
                if v[3 * j + 2] <= lam_p / lmax:
                    pat.append([px, py])
            sims.append(pat)
    return DescriptiveResult(
        name="poisson_process_fit",
        value=theta,
        extra={
            "se": se,
            "loglik": loglik,
            "expected_count": ssum(w[i] * lam[i] for i in range(len(quad))),
            "simulated": sims,
        },
    )


poifit = poisson_process_fit


def cheatsheet() -> str:
    return "poisson_process_fit(points, window) -> log-linear Poisson process MLE (Berman-Turner) and simulation"
