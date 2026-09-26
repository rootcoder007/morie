"""Tests for morie.fn.geods -- geodesic equation solver."""

from morie.fn import _array_core as np
import pytest

from morie.fn.geods import geods


def flat_metric(x):
    return np.diag([-1.0, 1.0, 1.0, 1.0])


def test_returns_dict():
    r = geods(flat_metric, np.array([0, 1, 0, 0]), np.array([1, 0.5, 0, 0]), tau_span=(0, 1), n_points=20)
    assert isinstance(r, dict)
    for k in ("tau", "position", "velocity"):
        assert k in r


def test_flat_space_straight_line():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    u0 = np.array([1.0, 0.5, 0.0, 0.0])
    r = geods(flat_metric, x0, u0, tau_span=(0, 2), n_points=50)
    # zero Christoffel symbols: x(tau) = x0 + u0 tau exactly; RK45 at
    # rtol 1e-10 reproduces it to integration accuracy
    for tau, pos in zip(r["tau"].tolist(), r["position"].tolist()):
        assert pos == pytest.approx([1.0 * tau, 0.5 * tau, 0.0, 0.0], abs=1e-8)


def test_velocity_conservation_flat():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    u0 = np.array([1.0, 0.3, 0.2, 0.0])
    r = geods(flat_metric, x0, u0, tau_span=(0, 1), n_points=30)
    v_init = r["velocity"][0]
    v_final = r["velocity"][-1]
    np.testing.assert_allclose(v_init, v_final, atol=1e-6)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        geods(flat_metric, np.zeros(3), np.zeros(4))


def test_straight_line_in_spherical_coordinates():
    """Flat space in spherical coordinates, g = diag(-1, 1, r^2, r^2
    sin^2 theta), has non-zero Christoffel symbols; starting at r = 1 on
    the equator with d phi / d tau = 1 the geodesic is the Cartesian line
    (1, tau), i.e. r = sqrt(1 + tau^2), phi = atan(tau), theta = pi/2.
    Central differences of the metric (h = 1e-5, error O(h^2)) and RK45
    at rtol 1e-10 bound the error well below 1e-6."""
    import math

    def sph(x):
        r, th = x[1], x[2]
        return np.diag([-1.0, 1.0, r * r, (r * math.sin(th)) ** 2])

    out = geods(sph, np.array([0.0, 1.0, math.pi / 2, 0.0]),
                np.array([1.0, 0.0, 0.0, 1.0]), tau_span=(0, 2), n_points=21)
    for tau, pos in zip(out["tau"].tolist(), out["position"].tolist()):
        assert pos[1] == pytest.approx(math.sqrt(1 + tau * tau), abs=1e-6)
        assert pos[3] == pytest.approx(math.atan(tau), abs=1e-6)
        assert pos[2] == pytest.approx(math.pi / 2, abs=1e-9)
