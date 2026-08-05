"""Ripley's K function for a mapped point pattern in a rectangle."""

import math as _math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ripley_k_function"]


def _rect(window):
    w = [float(v) for v in np.asarray(window, dtype=float).ravel().tolist()]
    if len(w) != 4:
        raise ValueError("`window` must be (xmin, xmax, ymin, ymax)")
    x0, x1, y0, y1 = w
    if not (x1 > x0 and y1 > y0):
        raise ValueError("`window` must have xmax > xmin and ymax > ymin")
    return x0, x1, y0, y1


def isotropic_weight(x, y, rad, x0, x1, y0, y1):
    """Proportion of the circle centred at ``(x, y)`` with radius ``rad``
    that lies inside the rectangle ``[x0, x1] x [y0, y1]``.

    This is Ripley's edge-correction weight ``w(s_i, s_j)`` -- "the
    proportion of the circumference of a circle that is within the study
    region, centered at s_i, and with radius h_ij" (Schabenberger &
    Gotway 2005, Sec. 3.4.2, p. 102; the estimator is Ripley 1976).

    Closed form.  Let ``d_L, d_R, d_B, d_T`` be the distances from the
    centre to the four sides and ``a_k = arccos(min(d_k / rad, 1))``.
    The arc lying beyond side ``k`` has angular measure ``2 a_k``.  Two
    opposite sides can never both be crossed by the same angle, so the
    only overlaps are the four corners, and there are no triple
    intersections; inclusion-exclusion is therefore exact:

        |A_i n A_j| = max(0, a_i + a_j - pi/2)          (adjacent i, j)
        outside     = sum_k 2 a_k - sum_{adj} |A_i n A_j|
        w           = 1 - outside / (2 pi)

    (The corner term is derived by intersecting ``cos t <= -d_i/rad``
    with ``sin t <= -d_j/rad``, whose overlap has length
    ``arccos(u) - arcsin(v) = arccos(u) + arccos(v) - pi/2``.)
    """
    if rad <= 0.0:
        return 1.0
    d = [x - x0, x1 - x, y - y0, y1 - y]
    if min(d) < 0.0:
        raise ValueError("point lies outside `window`")
    a = [_math.acos(min(dk / rad, 1.0)) for dk in d]
    outside = 2.0 * (a[0] + a[1] + a[2] + a[3])
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
        v = a[i] + a[j] - 0.5 * _math.pi
        if v > 0.0:
            outside -= v
    w = 1.0 - outside / (2.0 * _math.pi)
    return w if w > 1e-12 else 1e-12


def ripley_k_function(points, window, r):
    """
    Ripley's K function for point patterns.

    ``lambda K(h)`` is the expected number of further events within
    distance ``h`` of an arbitrary event.  Two estimators are returned.

    Isotropic (Ripley 1976; Schabenberger & Gotway 2005, eq. on p. 102)::

        Ehat(h) = (1/n) sum_i sum_{j != i} w(s_i, s_j)^{-1} I(h_ij <= h)
        Khat(h) = lambdahat^{-1} Ehat(h),   lambdahat = n / area

    Reduced-sample / border correction (same section, p. 102).  The book
    prints the numerator indicator as ``d_j > h``; that is a misprint --
    the reduced sample conditions on the *centre* event being at least
    ``h`` from the boundary, so this implementation uses ``d_i > h``,
    which is the standard form and the one that is unbiased::

        Khat_b(h) = sum_{i: d_i > h} #{j != i: h_ij <= h}
                    / (lambdahat * #{i: d_i > h})

    Under complete spatial randomness ``K(h) = pi h^2``; the Besag
    transform ``L(h) = sqrt(K(h)/pi)`` is returned alongside so the CSR
    reference is the straight line ``L(h) = h``.

    Parameters
    ----------
    points : array-like, shape (n, 2)
        Event coordinates, all inside ``window``.
    window : sequence of 4 floats
        Rectangle ``(xmin, xmax, ymin, ymax)``.
    r : float or array-like
        Distances at which to evaluate K.

    Returns
    -------
    RichResult
        Payload keys: ``r``, ``k``, ``k_border``, ``l``, ``csr``,
        ``lambda_hat``, ``area``, ``n``, ``method``.

    References
    ----------
    Ripley, B. D. (1976). The second-order analysis of stationary point
    processes. Journal of Applied Probability 13(2), 255-266.
    Schabenberger, O. and Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis, Sec. 3.4.1-3.4.2, pp. 101-103.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("`points` must have shape (n, 2)")
    n = int(P.shape[0])
    if n < 2:
        raise ValueError("`points` needs at least 2 events")
    x0, x1, y0, y1 = _rect(window)
    rs = [float(v) for v in np.atleast_1d(np.asarray(r, dtype=float)).tolist()]
    if min(rs) < 0.0:
        raise ValueError("`r` must be non-negative")
    area = (x1 - x0) * (y1 - y0)
    lam = n / area

    px = [float(v) for v in P[:, 0].tolist()]
    py = [float(v) for v in P[:, 1].tolist()]
    for i in range(n):
        if not (x0 <= px[i] <= x1 and y0 <= py[i] <= y1):
            raise ValueError("every point must lie inside `window`")
    bdist = [min(px[i] - x0, x1 - px[i], py[i] - y0, y1 - py[i])
             for i in range(n)]

    d = [[0.0] * n for _ in range(n)]
    wt = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = _math.sqrt((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2)
            d[i][j] = dij
            wt[i][j] = isotropic_weight(px[i], py[i], dij, x0, x1, y0, y1)

    kiso, kbor, lvals, csr = [], [], [], []
    for h in rs:
        acc = 0.0
        for i in range(n):
            for j in range(n):
                if i != j and d[i][j] <= h:
                    acc += 1.0 / wt[i][j]
        kh = area * acc / (n * n)
        kiso.append(kh)
        lvals.append(_math.sqrt(kh / _math.pi) if kh > 0.0 else 0.0)
        csr.append(_math.pi * h * h)
        m = 0
        cnt = 0.0
        for i in range(n):
            if bdist[i] > h:
                m += 1
                for j in range(n):
                    if i != j and d[i][j] <= h:
                        cnt += 1.0
        kbor.append(cnt / (lam * m) if m > 0 else float("nan"))

    return RichResult(
        payload={
            "r": rs,
            "k": kiso,
            "k_border": kbor,
            "l": lvals,
            "csr": csr,
            "lambda_hat": lam,
            "area": area,
            "n": n,
            "method": "Ripley's K function (isotropic + border correction)",
        }
    )


def cheatsheet():
    return "ripk: Ripley's K function for point patterns"


# compact alias per ledger/NAMING.md
ripleyk = ripley_k_function
