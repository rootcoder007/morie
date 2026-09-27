"""Majority-rule geometry in two dimensions: the yolk, Plott's radial symmetry and the core.

McKelvey, R. D. (1986). Covering, dominance, and institution-free properties of social choice.
American Journal of Political Science 30, 283-314 (the yolk). Stone, R. E. and Tovey, C. A.
(1992). Limiting median lines do not suffice to determine the yolk. Social Choice and Welfare
9, 33-35. Plott, C. R. (1967). A notion of equilibrium and its possibility under majority rule.
American Economic Review 57, 787-806.
"""

import math

from ._richresult import RichResult
from .neldmd import nelder_mead

__all__ = ["yolk"]


def _crit_angles(P):
    n = len(P)
    ang = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
            if dx or dy:
                a = math.atan2(dx, -dy) % math.pi  # normals perpendicular to the pair
                ang.add(a)
    return sorted(ang)


def _median_proj(P, u):
    pr = sorted((p[0] * u[0] + p[1] * u[1], k) for k, p in enumerate(P))
    n = len(P)
    return pr[(n - 1) // 2] if n % 2 else None


def _radius(c, P, angles):
    """max over directions of the distance from c to the median line (n odd)."""
    best = 0.0
    arcs = angles + [angles[0] + math.pi] if angles else [0.0, math.pi]
    for a, b in zip(arcs, arcs[1:]):
        mid = 0.5 * (a + b)
        _, k = _median_proj(P, (math.cos(mid), math.sin(mid)))
        v = (c[0] - P[k][0], c[1] - P[k][1])
        for t in (a, b):
            best = max(best, abs(v[0] * math.cos(t) + v[1] * math.sin(t)))
        nv = math.hypot(*v)
        if nv > 0:
            phi = math.atan2(v[1], v[0]) % math.pi  # direction of v (or -v) as a normal angle
            for cand in (phi, phi + math.pi, phi - math.pi):
                if a <= cand <= b:
                    best = max(best, nv)
    return best


def yolk(ideals, tol=1e-12):
    r"""Yolk: the smallest disc meeting every median line of an odd set of ideal points in the plane.

    For a direction u the median line {z: z.u = med_i x_i.u} is unique when n is odd. The
    distance r(c) from c to the farthest median line is a maximum of |(c - x_k(u)).u| over
    directions, convex in c; the median voter k(u) changes only at directions perpendicular to
    a pair of ideal points, so each arc contributes its endpoint values or, when the direction
    of c - x_k falls inside it, ||c - x_k|| (the case where limiting median lines alone do not
    determine the yolk; Stone and Tovey 1992). r(c) is minimised by Nelder-Mead from the
    centroid; the core exists (radius 0) exactly when Plott's radial symmetry holds.

    Parameters
    ----------
    ideals : list of (x, y), odd count
    tol : float

    Returns
    -------
    RichResult
        Keys: center, radius, core (bool: radius <= tol).

    References
    ----------
    McKelvey, R. D. (1986). American Journal of Political Science 30, 283-314.
    Stone, R. E. and Tovey, C. A. (1992). Social Choice and Welfare 9, 33-35.

    Examples
    --------
    >>> r = yolk([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
    >>> round(r["radius"], 9), r["core"]
    (0.0, True)
    """
    P = [(float(x), float(y)) for x, y in ideals]
    n = len(P)
    if n % 2 == 0 or n < 3:
        raise ValueError("the yolk here needs an odd number (>= 3) of ideal points")
    angles = _crit_angles(P)
    c0 = [sum(p[0] for p in P) / n, sum(p[1] for p in P) / n]
    best = None
    for start in [c0] + [list(p) for p in P]:
        r = nelder_mead(lambda c: _radius(c, P, angles), start, step=0.25, xtol=1e-13, ftol=1e-15, max_iter=4000)
        if best is None or r["fun"] < best[1]:
            best = (r["x"], r["fun"])
    return RichResult(
        title="Yolk",
        summary_lines=[("radius", best[1])],
        payload={"center": best[0], "radius": best[1], "core": best[1] <= max(tol, 1e-9)},
    )


def cheatsheet():
    return "svgeom: exact yolk of an odd set of ideal points in the plane"
