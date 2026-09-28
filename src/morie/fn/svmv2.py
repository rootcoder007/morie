# morie.fn -- function file (rootcoder007/morie)
"""Median voter in 2D (Plott conditions)"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _depth(z, P, tol=1e-12):
    """Tukey (halfspace) depth of z: min over directions of the closed half-plane count."""
    ang = []
    same = 0
    for p in P:
        dx, dy = p[0] - z[0], p[1] - z[1]
        if abs(dx) <= tol and abs(dy) <= tol:
            same += 1
        else:
            ang.append(math.atan2(dy, dx))
    if not ang:
        return len(P)
    br = sorted({(a + s * math.pi / 2) % (2 * math.pi) for a in ang for s in (1, -1)})
    mids = [(br[i] + br[i + 1]) / 2 for i in range(len(br) - 1)] + [(br[-1] + br[0] + 2 * math.pi) / 2]
    best = len(P)
    for phi in mids:
        c = sum(1 for a in ang if math.cos(a - phi) >= 0)
        best = min(best, c)
    return best + same


def median_voter_2d(x, *, ideal_point=None):
    r"""Plott total-median test in two dimensions: is a point a majority-rule core (equilibrium)?

    ``x`` holds the voters' ideal points (rows). With Euclidean preferences a
    point ``z`` is undominated under simple majority iff every closed
    half-plane with ``z`` on its boundary contains at least ``n/2`` ideal
    points, i.e. its Tukey depth is at least ``n/2`` (Plott 1967; McKelvey
    and Schofield 1987). ``value`` is the depth of ``ideal_point``; without
    it, the deepest of the ideal points and the coordinate-wise median is
    tested. The depth is computed exactly by the angular sweep of Rousseeuw
    and Ruts (1996).

    References
    ----------
    Plott, C. R. (1967). A notion of equilibrium and its possibility under
    majority rule. *American Economic Review*, 57(4), 787-806.
    Rousseeuw, P. J. and Ruts, I. (1996). Algorithm AS 307: bivariate
    location depth. *Applied Statistics*, 45(4), 516-526.

    Examples
    --------
    >>> r = median_voter_2d([[0, 0], [1, 1], [2, 2], [0, 2], [2, 0]])
    >>> r.extra["point"], r.value, r.extra["is_core"]
    ([1.0, 1.0], 3, True)
    >>> median_voter_2d([[0, 0], [1, 0], [0, 1]]).extra["is_core"]
    False
    """
    X = x.tolist() if hasattr(x, "tolist") else list(x)
    if X and not isinstance(X[0], (list, tuple)):
        X = [X]
    P = [[float(a) for a in r] for r in X]
    n = len(P)
    if ideal_point is not None:
        z = [float(a) for a in (ideal_point.tolist() if hasattr(ideal_point, "tolist") else ideal_point)]
    else:
        med = []
        for k in range(2):
            s = sorted(p[k] for p in P)
            med.append(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
        z = max([med] + P, key=lambda c: _depth(c, P))
    d = _depth(z, P)
    return DescriptiveResult(name="svmv2", value=d, extra={"point": z, "is_core": d >= n / 2, "n": n})


medi = median_voter_2d


def cheatsheet() -> str:
    return "median_voter_2d(ideals, ideal_point) -> Tukey depth and Plott majority-core test"


# compact alias per ledger/NAMING.md
medianvoter2d = median_voter_2d
