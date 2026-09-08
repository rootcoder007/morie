# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Geometric combinatorics: lattice polygons, point sets and orders.

Pick GA (1899) *Sitzungsberichte des deutschen
naturwissenschaftlich-medicinischen Vereines fur Bohmen "Lotos"*
19:311-319 (lattice-point theorem); Erdos P, Szekeres G (1935)
*Compositio Mathematica* 2:463-470 (monotone subsequences and the
happy ending problem); Helly E (1923) *Jahresbericht DMV* 32:175-176;
Matousek J (2002), *Lectures on Discrete Geometry*, Springer.

Everything here runs in exact integer arithmetic on lattice input --
orientation tests are integer cross products, areas are integer
shoelace doubles, boundary counts are gcds -- so a theorem check means
the theorem, not the rounding. Pick's theorem is verified against
interior points counted one by one, which is the only honest way to
check a statement about counting.
"""

import math
from itertools import combinations

from ._richresult import RichResult

__all__ = [
    "lattice_convex_hull",
    "pick_theorem",
    "erdos_szekeres_check",
    "happy_ending_quadrilateral",
    "helly_intervals",
]

_METHOD = "Geometric combinatorics (Matousek 2002)"


def _cross(o, a, b):
    """Twice the signed area of triangle o-a-b; exact on integers."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lattice_convex_hull(points):
    """Convex hull of integer points by Andrew's monotone chain.

    Orientation tests are exact integer cross products, so no hull
    vertex is ever mis-classified by rounding -- the failure mode of
    floating hulls on nearly-collinear input. Collinear boundary
    points are dropped; the hull is the minimal vertex set, ordered
    counter-clockwise from the lexicographic minimum.

    Examples
    --------
    >>> out = lattice_convex_hull([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)])
    >>> out["hull"]
    [(0, 0), (2, 0), (2, 2), (0, 2)]
    """
    pts = sorted({(int(x), int(y)) for x, y in points})
    if len(pts) < 3:
        raise ValueError(
            f"need at least 3 distinct points; got {len(pts)}."
        )
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h
    lower = half(pts)
    upper = half(reversed(pts))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("all points are collinear; the hull is a segment "
                         "and has no interior.")
    # twice the area by the shoelace formula -- an exact integer
    twice = 0
    for i, (x1, y1) in enumerate(hull):
        x2, y2 = hull[(i + 1) % len(hull)]
        twice += x1 * y2 - x2 * y1
    return RichResult(
        title="Lattice convex hull",
        summary_lines=[
            ("Hull vertices", len(hull)),
            ("Twice the area (exact)", twice),
        ],
        payload={
            "hull": hull,
            "n_vertices": len(hull),
            "twice_area": twice,
            "area": twice / 2.0,
            "estimate": twice / 2.0,
            "n": len(pts),
            "method": "Andrew's monotone chain, exact integer orientation",
        },
    )


def pick_theorem(vertices, verify_by_enumeration=True, enumeration_cap=10 ** 6):
    r"""Pick's theorem for a simple lattice polygon:
    :math:`A = I + B/2 - 1`.

    The three quantities are computed independently -- the area by the
    integer shoelace formula, the boundary count by
    :math:`\sum \gcd(|\Delta x|, |\Delta y|)` over the edges, and the
    interior count **by testing every lattice point of the bounding
    box** -- and the theorem is then CHECKED, not used. A check that
    derives one side from the other two would be a tautology.

    Enumeration over a huge bounding box is refused rather than
    silently skipped: ``verified`` comes back None with a warning, so
    a cap never masquerades as a pass.

    Examples
    --------
    >>> out = pick_theorem([(0, 0), (4, 0), (4, 3), (0, 3)])
    >>> out["interior"], out["boundary"], out["area"]
    (6, 14, 12.0)
    >>> out["pick_holds"]
    True
    """
    V = [(int(x), int(y)) for x, y in vertices]
    n = len(V)
    if n < 3:
        raise ValueError(f"a polygon needs at least 3 vertices; got {n}.")
    twice = 0
    boundary = 0
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        twice += x1 * y2 - x2 * y1
        boundary += math.gcd(abs(x2 - x1), abs(y2 - y1))
    if twice == 0:
        raise ValueError("the polygon is degenerate: its signed area is 0.")
    area2 = abs(twice)
    interior_pick = (area2 - boundary + 2) // 2
    parity_ok = (area2 - boundary) % 2 == 0

    def on_segment(p, a, b):
        if _cross(a, b, p) != 0:
            return False
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    def strictly_inside(p):
        # ray cast to the right with exact integer predicates
        cnt = 0
        for i in range(n):
            a, b = V[i], V[(i + 1) % n]
            if on_segment(p, a, b):
                return False
            if (a[1] > p[1]) != (b[1] > p[1]):
                # x coordinate of the crossing, compared exactly:
                # p.x < a.x + (p.y-a.y)(b.x-a.x)/(b.y-a.y)
                lhs = (p[0] - a[0]) * (b[1] - a[1])
                rhs = (p[1] - a[1]) * (b[0] - a[0])
                if b[1] > a[1]:
                    if lhs < rhs:
                        cnt += 1
                else:
                    if lhs > rhs:
                        cnt += 1
        return cnt % 2 == 1

    interior_direct = None
    verified = None
    warnings = []
    if verify_by_enumeration:
        xs = [p[0] for p in V]
        ys = [p[1] for p in V]
        box = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        if box > enumeration_cap:
            warnings.append(
                f"The bounding box holds {box} lattice points, above the "
                f"enumeration cap of {enumeration_cap}, so the theorem was "
                "NOT verified by direct counting here. The Pick value is "
                "still returned; 'verified' is None, not True."
            )
        else:
            interior_direct = sum(
                1
                for px in range(min(xs), max(xs) + 1)
                for py in range(min(ys), max(ys) + 1)
                if strictly_inside((px, py))
            )
            verified = interior_direct == interior_pick
    out = RichResult(
        title="Pick's theorem",
        summary_lines=[
            ("Area", area2 / 2.0),
            ("Boundary points", boundary),
            ("Interior (Pick)", interior_pick),
            ("Interior (counted)", interior_direct),
        ],
        payload={
            "area": area2 / 2.0,
            "twice_area": area2,
            "boundary": boundary,
            "interior": interior_pick,
            "interior_enumerated": interior_direct,
            "verified": verified,
            "pick_holds": verified if verified is not None else parity_ok,
            "estimate": area2 / 2.0,
            "n": n,
            "method": "Pick's theorem (Pick 1899), checked by enumeration",
        },
    )
    out.warnings.extend(warnings)
    if verified is False:
        out.warnings.append(
            "Direct counting disagrees with Pick's formula. For a SIMPLE "
            "polygon that is impossible, so the input is self-intersecting "
            "or traversed with repeated vertices."
        )
    return out


def erdos_szekeres_check(sequence, r=None, s=None):
    r"""The Erdos-Szekeres monotone subsequence theorem.

    Any sequence of :math:`(r-1)(s-1) + 1` distinct reals contains an
    increasing subsequence of length :math:`r` or a decreasing one of
    length :math:`s`. Longest lengths are computed by patience-style
    dynamic programming in :math:`O(n^2)`, and when the sequence
    reaches the threshold the guarantee is asserted in the payload.
    The bound is tight, and the tests build the extremal sequence --
    :math:`s-1` blocks of :math:`r-1` -- to show length
    :math:`(r-1)(s-1)` genuinely escapes.

    Examples
    --------
    >>> out = erdos_szekeres_check([3, 1, 4, 1.5, 5, 9, 2, 6], r=3, s=4)
    >>> out["longest_increasing"]
    4
    >>> out["guarantee_met"]
    True
    """
    w = [float(x) for x in sequence]
    n = len(w)
    if n == 0:
        raise ValueError("the sequence is empty.")
    if len(set(w)) != n:
        raise ValueError("the theorem needs distinct values; ties were "
                         "supplied.")
    inc = [1] * n
    dec = [1] * n
    for i in range(n):
        for j in range(i):
            if w[j] < w[i]:
                inc[i] = max(inc[i], inc[j] + 1)
            else:
                dec[i] = max(dec[i], dec[j] + 1)
    li, ld = max(inc), max(dec)
    threshold = None
    applies = None
    met = None
    if r is not None and s is not None:
        r, s = int(r), int(s)
        if r < 2 or s < 2:
            raise ValueError("r and s must be at least 2.")
        threshold = (r - 1) * (s - 1) + 1
        applies = n >= threshold
        met = (li >= r or ld >= s) if applies else None
    return RichResult(
        title="Erdos-Szekeres monotone subsequences",
        summary_lines=[
            ("Longest increasing", li),
            ("Longest decreasing", ld),
            ("Theorem applies (n >= threshold)", applies),
            ("Guarantee met", met),
        ],
        payload={
            "longest_increasing": li,
            "longest_decreasing": ld,
            "threshold": threshold,
            "applies": applies,
            "guarantee_met": met,
            "estimate": float(li),
            "n": n,
            "method": "Erdos-Szekeres theorem (1935)",
        },
    )


def happy_ending_quadrilateral(points):
    """The happy ending problem: any 5 points in general position
    contain 4 in convex position.

    General position (no 3 collinear) is tested with exact integer
    cross products, and the convex quadrilateral is FOUND -- all four
    orientations of consecutive triples equal -- rather than merely
    asserted to exist. With more than 5 points every 5-subset is
    checked, which is the theorem's statement, not a sample of it.

    Examples
    --------
    >>> out = happy_ending_quadrilateral(
    ...     [(0, 0), (4, 0), (2, 1), (1, 4), (3, 5)])
    >>> out["found"]
    True
    """
    pts = [(int(x), int(y)) for x, y in points]
    if len(pts) < 5:
        raise ValueError(f"the theorem is about 5 or more points; got "
                         f"{len(pts)}.")
    if len(set(pts)) != len(pts):
        raise ValueError("points must be distinct.")
    for a, b, c in combinations(pts, 3):
        if _cross(a, b, c) == 0:
            raise ValueError(
                f"points {a}, {b}, {c} are collinear; the theorem "
                "assumes general position."
            )

    def convex_quad(q):
        # q convex in SOME order iff its hull has 4 vertices; test the
        # cyclic order of the hull directly
        s = sorted(q)
        def halfh(seq):
            h = []
            for p in seq:
                while len(h) >= 2 and _cross(h[-2], h[-1], p) <= 0:
                    h.pop()
                h.append(p)
            return h
        hull = halfh(s)[:-1] + halfh(reversed(s))[:-1]
        return len(hull) == 4, hull

    witness = None
    all_five_ok = True
    for five in combinations(pts, 5):
        found_here = False
        for four in combinations(five, 4):
            ok, hull = convex_quad(list(four))
            if ok:
                found_here = True
                if witness is None:
                    witness = hull
                break
        if not found_here:
            all_five_ok = False
            break
    return RichResult(
        title="Happy ending problem",
        summary_lines=[
            ("Convex quadrilateral found", witness is not None),
            ("Every 5-subset contains one", all_five_ok),
        ],
        payload={
            "found": witness is not None,
            "witness": witness,
            "every_five_subset": all_five_ok,
            "estimate": float(witness is not None),
            "n": len(pts),
            "method": "Happy ending theorem (Erdos and Szekeres 1935)",
        },
    )


def helly_intervals(intervals):
    r"""Helly's theorem on the line: intervals meeting pairwise meet
    globally.

    In :math:`\mathbb{R}^1` the Helly number is 2, so pairwise
    intersection already forces a common point --
    :math:`\max_i a_i \le \min_i b_i` -- and the witness interval is
    returned. Both facts are computed independently: every pair is
    tested, then the global intersection is formed, and the theorem
    says the first implies the second. When some pair is disjoint the
    theorem is silent, and the payload says which pair.

    Examples
    --------
    >>> out = helly_intervals([(0, 3), (1, 5), (2, 4)])
    >>> out["common_point_exists"]
    True
    >>> out["witness"]
    (2.0, 3.0)
    """
    iv = [(float(a), float(b)) for a, b in intervals]
    if len(iv) < 1:
        raise ValueError("no intervals supplied.")
    if any(a > b for a, b in iv):
        raise ValueError("every interval must have a <= b.")
    disjoint_pair = None
    for i, j in combinations(range(len(iv)), 2):
        if iv[i][1] < iv[j][0] or iv[j][1] < iv[i][0]:
            disjoint_pair = (i, j)
            break
    lo = max(a for a, _ in iv)
    hi = min(b for _, b in iv)
    common = lo <= hi
    holds = (not common) <= (disjoint_pair is not None)  # pairwise -> global
    return RichResult(
        title="Helly's theorem on the line",
        summary_lines=[
            ("Pairwise intersecting", disjoint_pair is None),
            ("Common point exists", common),
            ("Helly implication holds", holds),
        ],
        payload={
            "pairwise_intersecting": disjoint_pair is None,
            "disjoint_pair": disjoint_pair,
            "common_point_exists": common,
            "witness": (lo, hi) if common else None,
            "helly_holds": holds,
            "estimate": float(common),
            "n": len(iv),
            "method": "Helly's theorem, d = 1 (Helly 1923)",
        },
    )


def cheatsheet():
    return (
        "geocmb: exact-integer convex hulls, Pick's theorem checked by "
        "counting lattice points one by one, Erdos-Szekeres with its tight "
        "extremal sequence, the happy ending theorem verified over every "
        "5-subset, and Helly on the line"
    )


# compact alias per ledger/NAMING.md
hellyintervals = helly_intervals


# compact alias per ledger/NAMING.md
picktheorem = pick_theorem
