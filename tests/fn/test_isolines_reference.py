"""isolines: analytic contours (circles, straight lines) and marching-squares invariants."""

import math

import pytest

from morie.fn.isolines import iso_lines


def test_plane_gives_straight_segments():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.5, 1.0]
    z = [[x + 2 * y for y in ys] for x in xs]
    r = iso_lines(xs, ys, z, [1.5, 2.2])
    assert len(r.lines) == 2
    for ln in r.lines:
        assert all(x + 2 * y == pytest.approx(ln["level"], abs=1e-12) for x, y in zip(ln["x"], ln["y"]))
    # level 2.2 runs from (2.2, 0) to (0.2, 1): length sqrt(4 + 1)
    assert r.length[1] == pytest.approx(math.sqrt(5), abs=1e-12)


def test_cone_contours_are_closed_polygons_inscribed_in_circles():
    g = [i * 0.25 - 2 for i in range(17)]
    z = [[math.hypot(x, y) for y in g] for x in g]
    r = iso_lines(g, g, z, [1.0, 1.6])
    for ln in r.lines:
        assert (ln["x"][0], ln["y"][0]) == (ln["x"][-1], ln["y"][-1])
        # vertices lie on grid edges, interpolated linearly between cone heights: radius within the chord error
        for x, y in zip(ln["x"], ln["y"]):
            assert abs(math.hypot(x, y) - ln["level"]) < 0.02
    assert r.length[0] == pytest.approx(2 * math.pi, rel=0.02)
    assert r.length[1] == pytest.approx(2 * math.pi * 1.6, rel=0.02)


def test_saddle_missing_and_errors():
    z = [[1.0, 0.0], [0.0, 1.0]]
    hi = iso_lines([0, 1], [0, 1], z, [0.4])  # centre 0.5 above: above-corners joined, below corners cut off
    assert len(hi.lines) == 2 and hi.length[0] == pytest.approx(2 * math.sqrt(2) * 0.4)
    lo = iso_lines([0, 1], [0, 1], z, [0.6])  # centre below: above corners cut off
    assert len(lo.lines) == 2 and lo.length[0] == pytest.approx(2 * math.sqrt(2) * 0.4)
    nan = float("nan")
    assert iso_lines([0, 1, 2], [0, 1], [[0, 0], [nan, 2], [2, 2]], [1.0]).length[0] >= 0
    assert iso_lines([0, 1], [0, 1], [[0, 0], [0, 0]], [1.0]).lines == []
    with pytest.raises(ValueError):
        iso_lines([0, 1], [0, 1, 2], [[0, 0], [0, 0]], [1.0])
