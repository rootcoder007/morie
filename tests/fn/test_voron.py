"""Tests for morie.fn.voron.

Areas are re-derived here by hand geometry (the exact unit-square cells at
the interior of a unit lattice) and by an independent shoelace sum over the
returned vertices, never by trusting the value the function reports.
"""

from morie.fn import _array_core as np

from morie.fn.voron import voron, voronoi_areas


def _shoelace(pts):
    """Shoelace area, written out term by term."""
    s = 0.0
    m = len(pts)
    for i in range(m):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _lattice(k, s=1.0):
    """Row-major k x k lattice with spacing s; point (i, j) is at index i*k+j."""
    return [[s * i, s * j] for i in range(k) for j in range(k)]


def test_voron_lattice_interior_cells_are_unit_squares():
    """On a unit 3x3 lattice the centre cell is exactly [.5,1.5]^2, area 1."""
    res = voron(points=_lattice(3))

    assert res.name == "Voronoi"
    assert res.extra["n_points"] == 9
    assert res.extra["n_finite"] + res.extra["n_infinite"] == 9

    areas = [float(a) for a in res.extra["areas"].tolist()]
    assert len(areas) == 9
    # The centre point (1, 1) is index 1*3+1 = 4.
    assert abs(areas[4] - 1.0) < 1e-12

    # Its cell really is the unit square, checked against the returned vertices.
    verts = [[float(c) for c in row] for row in res.extra["vertices"].tolist()]
    square = [(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)]
    unit = [
        r
        for r in res.extra["regions"]
        if sorted({(round(verts[k][0], 9), round(verts[k][1], 9)) for k in r}) == square
    ]
    assert len(unit) == 1
    assert abs(_shoelace([verts[k] for k in unit[0]]) - 1.0) < 1e-12

    # Every reported area is a real, non-negative number or NaN, never negative.
    for a in areas:
        assert a != a or a >= 0.0


def test_voron_scaling_squares_the_areas():
    """Scaling the lattice by s multiplies every cell area by s**2."""
    s = 2.5
    base = voron(points=_lattice(4))
    big = voron(points=_lattice(4, s))

    b = [float(a) for a in base.extra["areas"].tolist()]
    g = [float(a) for a in big.extra["areas"].tolist()]
    assert len(b) == len(g) == 16
    assert big.extra["n_finite"] == base.extra["n_finite"]

    for x, y in zip(b, g):
        if x == x:
            assert y == y
            assert abs(y - x * s * s) < 1e-9
        else:
            assert y != y
    assert abs(big.value - base.value * s * s) < 1e-9

    # The four interior points of a 4x4 unit lattice own unit squares.
    for i in (1, 2):
        for j in (1, 2):
            assert abs(b[i * 4 + j] - 1.0) < 1e-12
            assert abs(g[i * 4 + j] - s * s) < 1e-9


def test_voron_areas_match_an_independent_shoelace():
    """Each reported area equals a shoelace sum over the returned vertices."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(size=(20, 2))
    res = voronoi_areas(pts)

    verts = [[float(c) for c in row] for row in res.extra["vertices"].tolist()]
    regions = res.extra["regions"]
    areas = [float(a) for a in res.extra["areas"].tolist()]

    assert len(areas) == 20
    assert res.extra["n_finite"] + res.extra["n_infinite"] == 20
    assert res.extra["n_finite"] == sum(1 for a in areas if a == a)
    assert res.extra["n_finite"] > 0

    finite_regions = [r for r in regions if r and -1 not in r]
    hand = sorted(_shoelace([verts[k] for k in r]) for r in finite_regions)
    got = sorted(a for a in areas if a == a)
    assert len(hand) == len(got)
    for h, a in zip(hand, got):
        assert abs(h - a) < 1e-12

    assert abs(res.value - sum(got) / len(got)) < 1e-12


def test_voron_rejects_bad_shapes():
    """Wrong dimensionality and too-few points raise."""
    for bad in ([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0, 0.0]] * 5, [0.0, 1.0, 2.0]):
        try:
            voron(points=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))


def test_cheatsheet():
    from morie.fn.voron import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert "voronoi_areas" in cs
