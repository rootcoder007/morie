"""Tests for contours: isobands, clipping, band quantities, labels, smoothing."""

import math

from morie.fn.contours import (
    contour_bands,
    contour_clip,
    contour_fill,
    contour_labels,
    contour_quantity,
    contour_smooth,
)

XS = [0.0, 1.0, 2.0]
YS = [0.0, 1.0, 2.0]
LINEAR = [[x + 2 * y for x in XS] for y in YS]  # z = x + 2 y


def test_contour_fill_areas_are_exact_for_a_linear_field():
    # area where x + 2 y < c on [0, 2]^2 is c^2 / 4 for c <= 2
    r = contour_fill(LINEAR, XS, YS, [0.0, 1.0, 2.0, 6.5])
    assert abs(r.areas[0] - 1 / 4) <= 1e-12
    assert abs(r.areas[1] - (1 - 1 / 4)) <= 1e-12
    assert abs(sum(r.areas) - 4.0) <= 1e-12
    assert all(len(p) >= 3 for band in r.bands for p in band)


def test_contour_quantity_integral_of_a_linear_field():
    # int over {x + 2 y < 2} of (x + 2 y) = 4 / 3 on area 1
    r = contour_quantity(LINEAR, XS, YS, 0.0, 2.0)
    assert abs(r.area - 1.0) <= 1e-12
    assert abs(r.integral - 4 / 3) <= 1e-12
    assert abs(r.mean - 4 / 3) <= 1e-12


def test_contour_clip_pieces():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    r = contour_clip([[(-1, 0.5), (2, 0.5)], [(3, 3), (4, 4)], [(-1, -1), (2, 2)]], sq)
    assert r.n_pieces == 2
    assert r.lines[0] == [(0.0, 0.5), (1.0, 0.5)]
    (xa, ya), (xb, yb) = r.lines[1][0], r.lines[1][-1]
    assert abs(xa) <= 1e-12 and abs(ya) <= 1e-12 and abs(xb - 1) <= 1e-12 and abs(yb - 1) <= 1e-12


def test_contour_labels_pick_the_straightest_run():
    r = contour_labels([[(0, 0), (1, 0), (2, 0), (3, 1)], [(0, 0), (0.1, 0)]], min_length=0.5)
    assert r.positions == [(0.5, 0.0)] and r.angles == [0.0]
    # a tie in turning angle goes to the longest segment
    r2 = contour_labels([[(0, 0), (1, 0), (2, 1)]])
    assert r2.positions == [(1.5, 0.5)] and abs(r2.angles[0] - math.pi / 4) <= 1e-12


def test_contour_smooth_degree_one_reproduces_vertices_and_degree_two_bezier_midpoint():
    line = [(0, 0), (1, 1), (2, 0), (3, 1)]
    r = contour_smooth(line, degree=1, samples=4)
    for (x, y), (px, py) in zip(line, r.points):
        assert abs(x - px) <= 1e-12 and abs(y - py) <= 1e-12
    q = contour_smooth([(0, 0), (1, 1), (2, 0)], degree=2, samples=3)
    # quadratic Bezier at t = 1/2: P0/4 + P1/2 + P2/4
    assert abs(q.points[1][0] - 1.0) <= 1e-12 and abs(q.points[1][1] - 0.5) <= 1e-12
    c = contour_smooth([(0, 0), (1, 0), (1, 1), (0, 1)], degree=2, samples=8, closed=True)
    assert len(c.points) == 8 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in c.points)


def test_contour_bands_shades():
    r = contour_bands(LINEAR, XS, YS, [0.0, 2.0, 4.0, 6.5])
    assert [round(s, 12) for s in r.shades] == [0.9, 0.55, 0.2]
    assert r.areas == contour_fill(LINEAR, XS, YS, [0.0, 2.0, 4.0, 6.5]).areas
