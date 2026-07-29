"""Geometric combinatorics on exact integer arithmetic.

Pick's theorem is checked against interior points counted one by one;
the hull is checked by idempotence and by every input point lying
inside or on it; Erdos-Szekeres by building the extremal sequence that
escapes; the happy ending theorem over every 5-subset. The fuzz
polygons are star-shaped around the vertex centroid, which guarantees
simplicity -- an angular sort around a point OUTSIDE the hull produces
self-intersecting polygons, and Pick correctly rejects those.

Sources: Pick (1899); Erdos and Szekeres (1935); Helly (1923);
Matousek (2002) *Lectures on Discrete Geometry*.
"""

import math
import random
from fractions import Fraction
from itertools import combinations

import pytest

from morie.fn.geocmb import (
    erdos_szekeres_check,
    happy_ending_quadrilateral,
    helly_intervals,
    lattice_convex_hull,
    pick_theorem,
)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def star_polygon(rng, n_min=3, n_max=12, span=12):
    """A guaranteed-simple lattice polygon: angular sort around the
    centroid, same-ray duplicates dropped."""
    n = rng.randint(n_min, n_max)
    raw = set()
    while len(raw) < n:
        raw.add((rng.randint(-span, span), rng.randint(-span, span)))
    raw = list(raw)
    cx = Fraction(sum(p[0] for p in raw), len(raw))
    cy = Fraction(sum(p[1] for p in raw), len(raw))
    pts = sorted(raw, key=lambda p: (math.atan2(p[1] - cy, p[0] - cx),
                                     (p[0] - cx) ** 2 + (p[1] - cy) ** 2))
    keep, seen = [], set()
    for p in pts:
        dx, dy = p[0] - cx, p[1] - cy
        g = math.gcd(dx.numerator * dy.denominator,
                     dy.numerator * dx.denominator) or 1
        d = (dx.numerator * dy.denominator // g,
             dy.numerator * dx.denominator // g)
        if d not in seen:
            seen.add(d)
            keep.append(p)
    return keep


# --------------------------------------------------------------------
# Convex hull
# --------------------------------------------------------------------

def test_the_hull_of_a_square_with_an_interior_point():
    out = lattice_convex_hull([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)])
    assert out["hull"] == [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert out["twice_area"] == 8
    assert out["area"] == 4.0


def test_collinear_boundary_points_are_dropped():
    out = lattice_convex_hull([(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)])
    assert (1, 0) not in out["hull"]
    assert out["n_vertices"] == 4


def test_the_hull_is_idempotent_and_contains_every_point():
    rng = random.Random(11)
    for _ in range(150):
        pts = [(rng.randint(-20, 20), rng.randint(-20, 20))
               for _ in range(30)]
        try:
            hull = lattice_convex_hull(pts)["hull"]
        except ValueError:
            continue
        again = lattice_convex_hull(hull)["hull"]
        assert sorted(hull) == sorted(again)
        # every input point inside or on the hull: all cross products
        # against each directed edge are >= 0 (ccw hull)
        for p in pts:
            assert all(
                cross(hull[i], hull[(i + 1) % len(hull)], p) >= 0
                for i in range(len(hull))
            )


def test_hull_validation():
    with pytest.raises(ValueError, match="at least 3 distinct"):
        lattice_convex_hull([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="collinear"):
        lattice_convex_hull([(0, 0), (1, 1), (2, 2), (3, 3)])


# --------------------------------------------------------------------
# Pick's theorem
# --------------------------------------------------------------------

def test_pick_on_a_rectangle_is_textbook():
    out = pick_theorem([(0, 0), (4, 0), (4, 3), (0, 3)])
    assert out["area"] == 12.0
    assert out["boundary"] == 14
    assert out["interior"] == 6
    assert out["verified"] is True


def test_pick_on_the_smallest_triangle():
    # the fundamental triangle: area 1/2, no interior points
    out = pick_theorem([(0, 0), (1, 0), (0, 1)])
    assert out["twice_area"] == 1
    assert out["boundary"] == 3
    assert out["interior"] == 0
    assert out["verified"] is True


def test_pick_verified_by_enumeration_on_400_simple_polygons():
    rng = random.Random(7)
    tried = 0
    for _ in range(400):
        keep = star_polygon(rng)
        if len(keep) < 3:
            continue
        try:
            out = pick_theorem(keep)
        except ValueError:
            continue
        tried += 1
        assert out["verified"] is True, keep
    assert tried > 300


def test_a_self_intersecting_polygon_is_caught_not_passed():
    # a crossing quadrilateral -- the symmetric bowtie has signed area
    # 0 and dies on the degeneracy guard, so use one whose crossing
    # leaves nonzero signed area. Pick's formula and direct counting
    # disagree, and the disagreement is the DETECTION, not a bug
    out = pick_theorem([(0, 0), (4, 0), (4, 3), (2, -2)])
    assert out["verified"] is False
    assert any("self-intersecting" in w for w in out.warnings)


def test_the_enumeration_cap_reports_rather_than_passes():
    out = pick_theorem([(0, 0), (2000, 0), (2000, 2000), (0, 2000)],
                       enumeration_cap=100)
    assert out["verified"] is None
    assert any("NOT verified" in w for w in out.warnings)
    assert out["interior"] == 1999 ** 2


def test_orientation_does_not_matter():
    cw = pick_theorem([(0, 3), (4, 3), (4, 0), (0, 0)])
    ccw = pick_theorem([(0, 0), (4, 0), (4, 3), (0, 3)])
    assert cw["interior"] == ccw["interior"]
    assert cw["area"] == ccw["area"]


def test_pick_validation():
    with pytest.raises(ValueError, match="at least 3"):
        pick_theorem([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="degenerate"):
        pick_theorem([(0, 0), (1, 1), (2, 2)])


# --------------------------------------------------------------------
# Erdos-Szekeres
# --------------------------------------------------------------------

def test_monotone_lengths_on_a_worked_sequence():
    out = erdos_szekeres_check([3, 1, 4, 1.5, 5, 9, 2, 6], r=3, s=4)
    assert out["longest_increasing"] == 4
    assert out["longest_decreasing"] == 2
    assert out["applies"] is True
    assert out["guarantee_met"] is True


def test_the_guarantee_holds_on_every_permutation_at_threshold():
    # r = s = 3: every permutation of length 5 has inc 3 or dec 3
    from itertools import permutations
    for w in permutations(range(1, 6)):
        out = erdos_szekeres_check(list(w), r=3, s=3)
        assert out["guarantee_met"] is True


def test_the_extremal_sequence_escapes_at_length_below_threshold():
    # s-1 descending blocks of r-1 ascending: length (r-1)(s-1),
    # longest increasing r-1, longest decreasing s-1
    for r, s in [(3, 3), (4, 4), (3, 5), (5, 3)]:
        seq = []
        for block in range(s - 1, 0, -1):
            base = block * 100
            seq.extend(base + i for i in range(r - 1))
        out = erdos_szekeres_check(seq, r=r, s=s)
        assert len(seq) == (r - 1) * (s - 1)
        assert out["applies"] is False
        assert out["longest_increasing"] == r - 1
        assert out["longest_decreasing"] == s - 1


def test_the_guarantee_holds_on_random_sequences_at_threshold():
    rng = random.Random(3)
    for _ in range(200):
        r = rng.randint(2, 5)
        s = rng.randint(2, 5)
        n = (r - 1) * (s - 1) + 1
        seq = rng.sample(range(1000), n)
        out = erdos_szekeres_check(seq, r=r, s=s)
        assert out["guarantee_met"] is True


def test_erdos_szekeres_validation():
    with pytest.raises(ValueError, match="empty"):
        erdos_szekeres_check([])
    with pytest.raises(ValueError, match="distinct"):
        erdos_szekeres_check([1, 2, 1])
    with pytest.raises(ValueError, match="at least 2"):
        erdos_szekeres_check([1, 2, 3], r=1, s=3)


# --------------------------------------------------------------------
# Happy ending
# --------------------------------------------------------------------

def test_a_worked_configuration_yields_a_convex_quadrilateral():
    out = happy_ending_quadrilateral([(0, 0), (4, 0), (2, 1), (1, 4),
                                      (3, 5)])
    assert out["found"] is True
    assert len(out["witness"]) == 4


def test_every_5_subset_of_random_general_position_points():
    rng = random.Random(5)
    checked = 0
    for _ in range(150):
        pts = []
        while len(pts) < 6:
            p = (rng.randint(-15, 15), rng.randint(-15, 15))
            if p not in pts:
                pts.append(p)
        try:
            out = happy_ending_quadrilateral(pts)
        except ValueError:
            continue          # collinear triple; the theorem is silent
        checked += 1
        assert out["every_five_subset"] is True
    assert checked > 50


def test_the_witness_really_is_convex():
    out = happy_ending_quadrilateral([(0, 0), (4, 0), (2, 1), (1, 4),
                                      (3, 5)])
    w = out["witness"]
    signs = [cross(w[i], w[(i + 1) % 4], w[(i + 2) % 4]) for i in range(4)]
    assert all(s > 0 for s in signs) or all(s < 0 for s in signs)


def test_happy_ending_validation():
    with pytest.raises(ValueError, match="5 or more"):
        happy_ending_quadrilateral([(0, 0), (1, 0), (0, 1), (1, 1)])
    with pytest.raises(ValueError, match="collinear"):
        happy_ending_quadrilateral([(0, 0), (1, 1), (2, 2), (5, 0),
                                    (0, 5)])
    with pytest.raises(ValueError, match="distinct"):
        happy_ending_quadrilateral([(0, 0), (0, 0), (1, 4), (5, 1),
                                    (3, 3)])


# --------------------------------------------------------------------
# Helly on the line
# --------------------------------------------------------------------

def test_pairwise_intersection_forces_a_common_point():
    out = helly_intervals([(0, 3), (1, 5), (2, 4)])
    assert out["pairwise_intersecting"] is True
    assert out["common_point_exists"] is True
    assert out["witness"] == (2.0, 3.0)


def test_a_disjoint_pair_is_named():
    out = helly_intervals([(0, 1), (2, 3), (0, 5)])
    assert out["pairwise_intersecting"] is False
    assert out["disjoint_pair"] == (0, 1)
    assert out["common_point_exists"] is False


def test_the_implication_holds_on_random_families():
    rng = random.Random(9)
    saw_pairwise = 0
    for _ in range(500):
        iv = []
        for _ in range(rng.randint(2, 8)):
            a = rng.uniform(0, 10)
            iv.append((a, a + rng.uniform(0, 6)))
        out = helly_intervals(iv)
        assert out["helly_holds"] is True
        if out["pairwise_intersecting"]:
            saw_pairwise += 1
            assert out["common_point_exists"] is True
    assert saw_pairwise > 20


def test_touching_intervals_share_exactly_one_point():
    out = helly_intervals([(0, 2), (2, 5)])
    assert out["common_point_exists"] is True
    assert out["witness"] == (2.0, 2.0)


def test_helly_validation():
    with pytest.raises(ValueError, match="no intervals"):
        helly_intervals([])
    with pytest.raises(ValueError, match="a <= b"):
        helly_intervals([(3, 1)])
