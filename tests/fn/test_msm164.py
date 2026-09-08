"""Tests for msm164.hyperplane_side (MVSML Eq. 9.1)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.msm164 import hyperplane_side


def test_msm164_point_on_the_plane_evaluates_to_zero():
    """x1 + x2 + x3 = 3 holds at (1, 1, 1) with beta0 = -3."""
    r = hyperplane_side([1.0, 1.0, 1.0], beta=[1.0, 1.0, 1.0], beta0=-3.0)
    assert float(r["value"]) == pytest.approx(0.0, abs=1e-12)
    assert bool(r["on_plane"])


def test_msm164_sides_and_euclidean_distance():
    """Plane x = 0 in 2D (beta = (1, 0)): distance of (3, 7) is 3, sign
    separates the half-spaces."""
    r = hyperplane_side(np.array([[3.0, 7.0], [-2.0, 1.0]]), beta=[1.0, 0.0])
    np.testing.assert_allclose(r["value"], [3.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(r["distance"], [3.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(r["side"], [1.0, -1.0], atol=1e-12)


def test_msm164_distance_is_scale_invariant():
    """Scaling beta and beta0 together rescales f but not the geometric
    distance -- |f| / ||beta|| is what makes it a margin."""
    p = np.array([2.0, -1.0])
    a = hyperplane_side(p, beta=[1.0, 2.0], beta0=0.5)
    b = hyperplane_side(p, beta=[10.0, 20.0], beta0=5.0)
    assert float(a["distance"]) == pytest.approx(float(b["distance"]), rel=1e-12)
    assert float(b["value"]) == pytest.approx(10 * float(a["value"]), rel=1e-12)


def test_msm164_rejects_degenerate_input():
    with pytest.raises(ValueError, match="zero vector"):
        hyperplane_side([1.0, 2.0], beta=[0.0, 0.0])
    with pytest.raises(ValueError, match="coordinates"):
        hyperplane_side([1.0, 2.0, 3.0], beta=[1.0, 2.0])
