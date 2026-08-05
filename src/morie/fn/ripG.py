"""G function -- nearest-neighbour event-to-event distance distribution."""

import math as _math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ripley_g_function"]


def ripley_g_function(points, window, r):
    """
    G (point-to-point) nearest-neighbour distance distribution function.

    ``G(y) = P(nearest-neighbour distance <= y)``.  With ``h_i`` the
    distance from event ``i`` to the nearest other event, the raw
    empirical estimate is (Schabenberger & Gotway 2005, Sec. 3.3.4,
    p. 98)::

        Ghat(y) = #(h_i <= y) / n

    That estimator is biased downwards near the boundary, because an
    event's true nearest neighbour may lie outside the window.  The
    reduced-sample (border) correction restricts to events whose
    distance ``b_i`` to the boundary already exceeds ``y``, for which
    the observed nearest neighbour is provably the true one::

        Ghat_b(y) = #{i: h_i <= y and b_i > y} / #{i: b_i > y}

    Under complete spatial randomness with intensity ``lambda``,
    ``G(y) = 1 - exp(-lambda pi y^2)``, which is returned as ``csr``.

    Parameters
    ----------
    points : array-like, shape (n, 2)
    window : sequence of 4 floats
        Rectangle ``(xmin, xmax, ymin, ymax)``.
    r : float or array-like
        Distances at which to evaluate G.

    Returns
    -------
    RichResult
        Payload keys: ``r``, ``g``, ``g_border``, ``nn``, ``csr``,
        ``mean_nn``, ``lambda_hat``, ``n``, ``method``.

    References
    ----------
    Schabenberger, O. and Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis, Sec. 3.3.4, p. 98.
    Ripley, B. D. (1976). The second-order analysis of stationary point
    processes. Journal of Applied Probability 13(2), 255-266.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("`points` must have shape (n, 2)")
    n = int(P.shape[0])
    if n < 2:
        raise ValueError("`points` needs at least 2 events")
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

    nn = []
    for i in range(n):
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            dij = _math.sqrt((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2)
            if dij < best:
                best = dij
        nn.append(best)
    bdist = [min(px[i] - x0, x1 - px[i], py[i] - y0, y1 - py[i])
             for i in range(n)]

    lam = n / ((x1 - x0) * (y1 - y0))
    g, gb, csr = [], [], []
    for h in rs:
        g.append(sum(1 for v in nn if v <= h) / float(n))
        m = sum(1 for b in bdist if b > h)
        if m > 0:
            k = sum(1 for i in range(n) if bdist[i] > h and nn[i] <= h)
            gb.append(k / float(m))
        else:
            gb.append(float("nan"))
        csr.append(1.0 - _math.exp(-lam * _math.pi * h * h))

    return RichResult(
        payload={
            "r": rs,
            "g": g,
            "g_border": gb,
            "nn": nn,
            "csr": csr,
            "mean_nn": sum(nn) / float(n),
            "lambda_hat": lam,
            "n": n,
            "method": "G function (nearest-neighbour distances, border corrected)",
        }
    )


def cheatsheet():
    return "ripG: G nearest-neighbour distance distribution function"


# compact alias per ledger/NAMING.md
ripleyg = ripley_g_function
