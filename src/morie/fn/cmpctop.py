"""Shape compactness of a polygon: Polsby-Popper, Schwartzberg, Reock and convex-hull ratios.

Polsby, D. D. and Popper, R. D. (1991). The third criterion: compactness as a procedural
safeguard against partisan gerrymandering. Yale Law and Policy Review 9, 301-353.
Schwartzberg, J. E. (1966). Reapportionment, gerrymanders, and the notion of compactness.
Minnesota Law Review 50, 443-452. Reock, E. C. (1961). A note: measuring compactness as a
requirement of legislative apportionment. Midwest Journal of Political Science 5, 70-74.
"""

import math

from ._richresult import RichResult

__all__ = ["polygon_compactness"]


def _area_perimeter(P):
    a, per = 0.0, 0.0
    for k in range(len(P)):
        (x1, y1), (x2, y2) = P[k], P[(k + 1) % len(P)]
        a += x1 * y2 - x2 * y1
        per += math.hypot(x2 - x1, y2 - y1)
    return abs(a) / 2, per


def _hull(P):
    pts = sorted(set(P))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _circle2(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, math.hypot(a[0] - c[0], a[1] - c[1])


def _circle3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-300:
        return None
    ux = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    uy = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    return (ux, uy), math.hypot(a[0] - ux, a[1] - uy)


def _inside(circ, p):
    return math.hypot(p[0] - circ[0][0], p[1] - circ[0][1]) <= circ[1] * (1 + 1e-12) + 1e-12


def min_enclosing_circle(P):
    """Smallest enclosing circle of points (exact over hull-point pairs and triples)."""
    H = _hull(P)
    if len(H) == 1:
        return H[0], 0.0
    best = None
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            c = _circle2(H[i], H[j])
            if (best is None or c[1] < best[1]) and all(_inside(c, p) for p in H):
                best = c
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            for k in range(j + 1, len(H)):
                c = _circle3(H[i], H[j], H[k])
                if c is not None and (best is None or c[1] < best[1]) and all(_inside(c, p) for p in H):
                    best = c
    return best


def polygon_compactness(vertices):
    r"""Compactness scores of a simple polygon given by its vertices (either orientation).

    polsby_popper = 4 pi A / P^2; schwartzberg = 1 / (P / (2 sqrt(pi A))) (perimeter of the
    equal-area circle over the perimeter); reock = A / area of the minimum enclosing circle;
    convex_hull = A / area of the convex hull. All equal 1 for a disc and fall toward 0 for
    elongated or indented shapes.

    Parameters
    ----------
    vertices : list of (x, y)

    Returns
    -------
    RichResult
        Keys: area, perimeter, polsby_popper, schwartzberg, reock, convex_hull,
        enclosing_circle (centre, radius).

    References
    ----------
    Polsby, D. D. and Popper, R. D. (1991). Yale Law and Policy Review 9, 301-353.
    Schwartzberg, J. E. (1966). Minnesota Law Review 50, 443-452.
    Reock, E. C. (1961). Midwest Journal of Political Science 5, 70-74.

    Examples
    --------
    >>> round(polygon_compactness([(0, 0), (1, 0), (1, 1), (0, 1)])["polsby_popper"], 10)
    0.7853981634
    """
    P = [(float(x), float(y)) for x, y in vertices]
    if len(P) < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    A, per = _area_perimeter(P)
    if A <= 0:
        raise ValueError("polygon has zero area")
    H = _hull(P)
    Ah, _ = _area_perimeter(H)
    circ = min_enclosing_circle(P)
    return RichResult(
        title="Polygon compactness",
        summary_lines=[("Polsby-Popper", 4 * math.pi * A / per**2)],
        payload={
            "area": A,
            "perimeter": per,
            "polsby_popper": 4 * math.pi * A / per**2,
            "schwartzberg": 2 * math.sqrt(math.pi * A) / per,
            "reock": A / (math.pi * circ[1] ** 2),
            "convex_hull": A / Ah,
            "enclosing_circle": (list(circ[0]), circ[1]),
        },
    )


def cheatsheet():
    return "cmpctop: polygon compactness (Polsby-Popper, Schwartzberg, Reock, convex hull)"
