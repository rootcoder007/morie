import math

from morie.fn.geomops import elevation_profile, filled_contour_bands, polygon_area, polygon_boolean, proximity_bands

A = [(0, 0), (5, 0), (6, 3), (3, 5), (-1, 3)]
B = [(2, -1), (7, 2), (4, 6.5), (1, 2.5)]
C = [(0, 0), (6, 0), (6, 1), (1, 1), (1, 4), (6, 4), (6, 5), (0, 5)]
D = [(4, -1), (5, -1), (5, 6), (4, 6)]


def test_boolean_area_identities():
    for P, Q in ((A, B), (C, D), (D, C)):
        i = polygon_boolean(P, Q, "intersection").area
        u = polygon_boolean(P, Q, "union").area
        d = polygon_boolean(P, Q, "difference").area
        aP, aQ = abs(polygon_area(P)), abs(polygon_area(Q))
        assert abs(u - (aP + aQ - i)) < 1e-9 and abs(d - (aP - i)) < 1e-9
    assert abs(polygon_boolean(C, D, "intersection").area - 2.0) < 1e-12  # two unit squares
    holes = [r for r in polygon_boolean(C, D, "union").rings if polygon_area(r) < 0]
    assert len(holes) == 1 and abs(polygon_area(holes[0]) + 9.0) < 1e-12


def test_disjoint_and_nested():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    big = [(-1, -1), (3, -1), (3, 3), (-1, 3)]
    assert polygon_boolean(sq, big, "intersection").area == 1.0
    assert polygon_boolean(big, sq, "difference").area == 15.0
    far = [(5, 5), (6, 5), (6, 6)]
    assert polygon_boolean(sq, far, "intersection").area == 0.0 and polygon_boolean(sq, far, "union").area == 1.5


def test_filled_contours_of_a_plane():
    x = [0, 0.5, 1.0, 2.0]
    y = [0, 1]
    z = [[v for v in x] for _ in y]  # z = x
    r = filled_contour_bands(x, y, z, [0, 0.5, 1.2, 2.0])
    assert all(abs(a - b) < 1e-12 for a, b in zip(r.areas, [0.5, 0.7, 0.8]))


def test_profile_and_bands():
    x, y = [0, 1, 2], [0, 1, 2]
    z = [[a + 2 * b for a in x] for b in y]  # bilinear-exact plane
    r = elevation_profile(x, y, z, [(0, 0), (2, 2)], 0.5)
    assert all(abs(e - 3 * d / math.sqrt(2)) < 1e-12 for e, d in zip(r.elevation, r.distance))
    assert abs(r.ascent - 6.0) < 1e-12 and r.descent == 0.0
    b = proximity_bands([0.5, 1.5, 2.5, 3.5], [0.5], [1.0, 2.0], lines=[[(0, 0), (0, 1)]])
    assert b.band == [[0, 1, 2, 2]] and b.cells_per_band == [1, 1, 2]
