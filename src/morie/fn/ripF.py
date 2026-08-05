"""F function -- empty-space (point-to-nearest-event) distance function."""

import math as _math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ripley_f_function"]

_GRID = 20


def ripley_f_function(points, window, r):
    """
    F (empty-space) nearest-event distance function.

    ``F(y) = P(distance from an arbitrary location to the nearest event
    <= y)``.  The location distribution is uniform over the window; a
    *deterministic* ``20 x 20`` lattice of cell centres is used as the
    quadrature sample so that both language arms land on identical
    numbers rather than merely the same distribution::

        u_ab = (x0 + (a - 0.5) dx, y0 + (b - 0.5) dy),  dx = (x1-x0)/20
        Fhat(y) = #(min_i ||u - s_i|| <= y) / m,        m = 400

    Reduced-sample (border) correction, restricting to test locations
    whose distance ``b_u`` to the boundary exceeds ``y`` -- for those the
    observed nearest event is provably the true one::

        Fhat_b(y) = #{u: dist(u) <= y and b_u > y} / #{u: b_u > y}

    Under complete spatial randomness ``F`` and ``G`` coincide,
    ``F(y) = 1 - exp(-lambda pi y^2)``; ``F`` above and ``G`` below that
    curve indicates regularity, the reverse indicates clustering.  The
    ratio ``J(y) = (1 - G(y)) / (1 - F(y))`` is the standard summary of
    that comparison and is not computed here (see ``ripG``).

    Parameters
    ----------
    points : array-like, shape (n, 2)
    window : sequence of 4 floats
        Rectangle ``(xmin, xmax, ymin, ymax)``.
    r : float or array-like
        Distances at which to evaluate F.

    Returns
    -------
    RichResult
        Payload keys: ``r``, ``f``, ``f_border``, ``csr``, ``m``,
        ``lambda_hat``, ``n``, ``method``.

    References
    ----------
    Schabenberger, O. and Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis, Sec. 3.3.4, pp. 97-98 (distance-based CSR
    tests; the empty-space distances are the point-to-event companion of
    the event-to-event distances given there).
    Ripley, B. D. (1976). The second-order analysis of stationary point
    processes. Journal of Applied Probability 13(2), 255-266.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("`points` must have shape (n, 2)")
    n = int(P.shape[0])
    if n < 1:
        raise ValueError("`points` needs at least 1 event")
    w = [float(v) for v in np.asarray(window, dtype=float).ravel().tolist()]
    if len(w) != 4:
        raise ValueError("`window` must be (xmin, xmax, ymin, ymax)")
    x0, x1, y0, y1 = w
    if not (x1 > x0 and y1 > y0):
        raise ValueError("`window` must have xmax > xmin and ymax > ymin")
    rs = [float(v) for v in np.atleast_1d(np.asarray(r, dtype=float)).tolist()]
    if min(rs) < 0.0:
        raise ValueError("`r` must be non-negative")

    px = [float(v) for v in P[:, 0].tolist()]
    py = [float(v) for v in P[:, 1].tolist()]
    for i in range(n):
        if not (x0 <= px[i] <= x1 and y0 <= py[i] <= y1):
            raise ValueError("every point must lie inside `window`")

    dx = (x1 - x0) / _GRID
    dy = (y1 - y0) / _GRID
    dmin, bmin = [], []
    for a in range(_GRID):
        ux = x0 + (a + 0.5) * dx
        for b in range(_GRID):
            uy = y0 + (b + 0.5) * dy
            best = float("inf")
            for i in range(n):
                dd = _math.sqrt((ux - px[i]) ** 2 + (uy - py[i]) ** 2)
                if dd < best:
                    best = dd
            dmin.append(best)
            bmin.append(min(ux - x0, x1 - ux, uy - y0, y1 - uy))
    m = _GRID * _GRID

    lam = n / ((x1 - x0) * (y1 - y0))
    f, fb, csr = [], [], []
    for h in rs:
        f.append(sum(1 for v in dmin if v <= h) / float(m))
        mm = sum(1 for b in bmin if b > h)
        if mm > 0:
            k = sum(1 for i in range(m) if bmin[i] > h and dmin[i] <= h)
            fb.append(k / float(mm))
        else:
            fb.append(float("nan"))
        csr.append(1.0 - _math.exp(-lam * _math.pi * h * h))

    return RichResult(
        payload={
            "r": rs,
            "f": f,
            "f_border": fb,
            "csr": csr,
            "m": m,
            "lambda_hat": lam,
            "n": n,
            "method": "F function (empty-space distances, 20x20 lattice)",
        }
    )


def cheatsheet():
    return "ripF: F empty-space nearest-event distance function"


# compact alias per ledger/NAMING.md
ripleyf = ripley_f_function
