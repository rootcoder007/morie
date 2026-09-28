# morie.fn -- function file (rootcoder007/morie)
"""Electoral geography: partisan-symmetry measures, swing, competitiveness, disproportionality,
malapportionment and district compactness."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "partisan_gerrymander_measures",
    "electoral_swing",
    "district_competitiveness",
    "disproportionality",
    "malapportionment",
    "district_compactness",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def partisan_gerrymander_measures(votes_a, votes_b) -> RichResult:
    r"""Efficiency gap, mean-median difference and partisan bias of a districting plan (two-party votes).

    - Efficiency gap (Stephanopoulos and McGhee 2015): wasted votes are all
      of a loser's votes and a winner's votes above half the district's
      two-party total; ``EG = (W_A - W_B) / sum T``, positive when party A
      wastes more (the plan favours B).
    - Mean-median difference (McDonald and Best 2015): mean minus median of
      A's district vote shares; positive when A's median district is below
      its mean (the plan favours B).
    - Partisan bias (uniform swing, Gelman and King 1994): shift every
      district share by ``0.5 - mean share`` and report A's seat share at a
      tied average minus 0.5.

    References
    ----------
    Stephanopoulos, N. O. and McGhee, E. M. (2015). Partisan gerrymandering
    and the efficiency gap. *University of Chicago Law Review*, 82, 831-900.
    McDonald, M. D. and Best, R. E. (2015). Unfair partisan gerrymanders in
    politics and law: a diagnostic applied to six cases. *Election Law
    Journal*, 14(4), 312-330.

    Examples
    --------
    >>> r = partisan_gerrymander_measures([70, 70, 40, 40, 40], [30, 30, 60, 60, 60])
    >>> round(r.efficiency_gap, 6), round(r.mean_median, 6), round(r.partisan_bias, 6)
    (0.14, 0.12, -0.1)
    """
    a, b = _vec(votes_a), _vec(votes_b)
    if len(a) != len(b):
        raise ValueError("votes_a and votes_b must have equal length")
    wa = wb = 0.0
    for x, y in zip(a, b):
        half = (x + y) / 2
        if x > y:
            wa += x - half
            wb += y
        else:
            wa += x
            wb += y - half
    tot = ssum(x + y for x, y in zip(a, b))
    sh = [x / (x + y) for x, y in zip(a, b)]
    m = ssum(sh) / len(sh)
    shift = 0.5 - m
    seats = sum(1 for s in sh if s + shift > 0.5) / len(sh)
    return RichResult(
        payload={
            "efficiency_gap": (wa - wb) / tot,
            "wasted_a": wa,
            "wasted_b": wb,
            "mean_median": m - _median(sh),
            "partisan_bias": seats - 0.5,
            "shares": sh,
        }
    )


def electoral_swing(prev_a, prev_b, cur_a, cur_b) -> RichResult:
    r"""Butler and Steed swing between two elections (inputs are vote shares or counts of each party's total).

    Butler swing ``((a2 - a1) - (b2 - b1))/2`` in shares of the total vote;
    Steed (two-party) swing ``a2/(a2 + b2) - a1/(a1 + b1)``. Positive values
    are swings towards A. With counts, pass shares of the total vote for the
    Butler figure.

    References
    ----------
    Butler, D. E. (1951). Appendix to H. G. Nicholas, *The British General
    Election of 1950*. Macmillan.
    Steed, M. (1965). An analysis of the results. In D. E. Butler and A. King,
    *The British General Election of 1964*. Macmillan.

    Examples
    --------
    >>> r = electoral_swing([0.40], [0.45], [0.46], [0.41])
    >>> round(r.butler[0], 6), round(r.steed[0], 6)
    (0.05, 0.058147)
    """
    a1, b1, a2, b2 = _vec(prev_a), _vec(prev_b), _vec(cur_a), _vec(cur_b)
    butler = [((z - x) - (w - y)) / 2 for x, y, z, w in zip(a1, b1, a2, b2)]
    steed = [z / (z + w) - x / (x + y) for x, y, z, w in zip(a1, b1, a2, b2)]
    return RichResult(payload={"butler": butler, "steed": steed})


def district_competitiveness(votes_a, votes_b, *, threshold: float = 0.05) -> RichResult:
    r"""Two-party margins ``|s - (1 - s)|`` per district and the districts within ``threshold`` of a tie (``|s - 0.5| < t``).

    Examples
    --------
    >>> r = district_competitiveness([52, 70, 48], [48, 30, 52])
    >>> [round(v, 6) for v in r.margin], r.n_competitive
    ([0.04, 0.4, 0.04], 2)
    """
    a, b = _vec(votes_a), _vec(votes_b)
    sh = [x / (x + y) for x, y in zip(a, b)]
    mg = [abs(2 * s - 1) for s in sh]
    comp = [abs(s - 0.5) < threshold for s in sh]
    return RichResult(
        payload={"margin": mg, "competitive": comp, "n_competitive": sum(comp), "mean_margin": ssum(mg) / len(mg)}
    )


def disproportionality(votes, seats) -> RichResult:
    r"""Vote-seat disproportionality and effective numbers of parties.

    Loosemore-Hanby ``D = (1/2) sum |v_i - s_i|``; Gallagher least squares
    ``LSq = sqrt((1/2) sum (v_i - s_i)^2)`` (in percentage points); effective
    number of parties ``1/sum p_i^2`` for votes and seats (Laakso and
    Taagepera 1979).

    References
    ----------
    Gallagher, M. (1991). Proportionality, disproportionality and electoral
    systems. *Electoral Studies*, 10(1), 33-51.
    Laakso, M. and Taagepera, R. (1979). "Effective" number of parties.
    *Comparative Political Studies*, 12(1), 3-27.

    Examples
    --------
    >>> r = disproportionality([40, 35, 25], [55, 40, 5])
    >>> round(r.loosemore_hanby, 6), round(r.gallagher, 6), round(r.enp_votes, 6)
    (20.0, 18.027756, 2.898551)
    """
    v, s = _vec(votes), _vec(seats)
    tv, ts = ssum(v), ssum(s)
    pv = [100 * x / tv for x in v]
    ps = [100 * x / ts for x in s]
    return RichResult(
        payload={
            "loosemore_hanby": ssum(abs(x - y) for x, y in zip(pv, ps)) / 2,
            "gallagher": math.sqrt(ssum((x - y) ** 2 for x, y in zip(pv, ps)) / 2),
            "enp_votes": 1 / ssum((x / 100) ** 2 for x in pv),
            "enp_seats": 1 / ssum((x / 100) ** 2 for x in ps),
        }
    )


def malapportionment(seats, population) -> RichResult:
    r"""Samuels and Snyder (2001) malapportionment ``MAL = (1/2) sum |s_i - p_i|`` and each unit's representation ratio ``s_i/p_i``.

    References
    ----------
    Samuels, D. and Snyder, R. (2001). The value of a vote: malapportionment
    in comparative perspective. *British Journal of Political Science*,
    31(4), 651-671.

    Examples
    --------
    >>> r = malapportionment([2, 2, 1], [100, 300, 100])
    >>> round(r.mal, 6), [round(v, 6) for v in r.ratio]
    (0.2, [2.0, 0.666667, 1.0])
    """
    s, p = _vec(seats), _vec(population)
    ts, tp = ssum(s), ssum(p)
    ss, pp = [x / ts for x in s], [x / tp for x in p]
    return RichResult(
        payload={"mal": ssum(abs(x - y) for x, y in zip(ss, pp)) / 2, "ratio": [x / y for x, y in zip(ss, pp)]}
    )


def _hull(P):
    P = sorted(set(P))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _area(P):
    n = len(P)
    return abs(ssum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))) / 2


def _circ2(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, math.dist(a, b) / 2


def _circ3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if d == 0:
        pts = [a, b, c]
        best = max(((p, q) for i, p in enumerate(pts) for q in pts[i + 1 :]), key=lambda t: math.dist(*t))
        return _circ2(*best)
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
    return (ux, uy), math.dist((ux, uy), a)


def _min_circle(P):
    eps = 1e-12

    def inside(c, p):
        return math.dist(c[0], p) <= c[1] * (1 + eps) + eps

    c = (P[0], 0.0)
    for i in range(1, len(P)):
        if inside(c, P[i]):
            continue
        c = (P[i], 0.0)
        for j in range(i):
            if inside(c, P[j]):
                continue
            c = _circ2(P[i], P[j])
            for k in range(j):
                if not inside(c, P[k]):
                    c = _circ3(P[i], P[j], P[k])
    return c


def district_compactness(polygon) -> RichResult:
    r"""Compactness of a simple polygon district (vertex list, no repeated closing vertex).

    Polsby-Popper ``4 pi A / P^2``; Schwartzberg ``P / (2 sqrt(pi A))``
    (perimeter over the circumference of the equal-area circle, >= 1);
    convex-hull ratio ``A / A_hull``; Reock ``A / (pi R^2)`` with ``R`` the
    radius of the minimum enclosing circle (Welzl's incremental algorithm on
    the hull vertices).

    References
    ----------
    Polsby, D. D. and Popper, R. D. (1991). The third criterion:
    compactness as a procedural safeguard against partisan gerrymandering.
    *Yale Law and Policy Review*, 9(2), 301-353.
    Reock, E. C. (1961). A note: measuring compactness as a requirement of
    legislative apportionment. *Midwest Journal of Political Science*, 5(1), 70-74.

    Examples
    --------
    >>> r = district_compactness([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> round(r.polsby_popper, 6), round(r.reock, 6), r.convex_hull
    (0.785398, 0.63662, 1.0)
    """
    P = [(float(x), float(y)) for x, y in polygon]
    A = _area(P)
    per = ssum(math.dist(P[i], P[(i + 1) % len(P)]) for i in range(len(P)))
    H = _hull(P)
    _, R = _min_circle(H)
    return RichResult(
        payload={
            "polsby_popper": 4 * math.pi * A / per**2,
            "schwartzberg": per / (2 * math.sqrt(math.pi * A)),
            "convex_hull": A / _area(H),
            "reock": A / (math.pi * R * R),
            "area": A,
            "perimeter": per,
        }
    )


def cheatsheet() -> str:
    return "partisan_gerrymander_measures / electoral_swing / disproportionality / district_compactness -> elections."
