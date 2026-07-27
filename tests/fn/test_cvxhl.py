"""Tests for cvxhl.convex_hull."""

import numpy as np
import pytest

from morie.fn.cvxhl import convex_hull


def _shoelace(poly):
    """Polygon area from ordered vertices, so the hull's own ordering is tested."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_cvxhl_ignores_interior_points():
    """The unit square with a point at its centre: the hull is the four
    corners, and the interior point must not survive."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    r = convex_hull(pts)
    hull = np.asarray(r.extra["hull_points"], dtype=float)
    assert int(r.value) == 4
    assert len(hull) == 4
    assert not any(np.allclose(v, [0.5, 0.5]) for v in hull)


def test_cvxhl_vertices_come_back_in_polygon_order():
    """Area by the shoelace formula only equals 1 if the vertices are returned
    in traversal order. A correct vertex set in scrambled order fails this."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.3, 0.7]])
    hull = np.asarray(convex_hull(pts).extra["hull_points"], dtype=float)
    assert _shoelace(hull) == pytest.approx(1.0, rel=1e-9)


def test_cvxhl_triangle_is_its_own_hull():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    r = convex_hull(pts)
    hull = np.asarray(r.extra["hull_points"], dtype=float)
    assert int(r.value) == 3
    # Area = 1/2 * base * height = 6.
    assert _shoelace(hull) == pytest.approx(6.0, rel=1e-9)


def test_cvxhl_many_interior_points_leave_the_hull_unchanged():
    """Scatter 200 points strictly inside the square; the hull stays the four
    corners. This is what a hull is for, and a bounding-box or
    nearest-neighbour stub would not reproduce it."""
    rng = np.random.default_rng(0)
    inner = rng.uniform(0.05, 0.95, size=(200, 2))
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hull = np.asarray(convex_hull(np.vstack([corners, inner])).extra["hull_points"], dtype=float)
    assert len(hull) == 4
    assert _shoelace(hull) == pytest.approx(1.0, rel=1e-9)


def test_cvxhl_indices_point_back_at_the_input():
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
    r = convex_hull(pts)
    idx = np.asarray(r.extra["hull_indices"], dtype=int)
    np.testing.assert_allclose(pts[idx], np.asarray(r.extra["hull_points"], dtype=float), atol=1e-12)


def test_cvxhl_rejects_wrong_shape_and_too_few_points():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        convex_hull(np.arange(10.0))
    with pytest.raises(ValueError, match="Need >= 3"):
        convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]]))
