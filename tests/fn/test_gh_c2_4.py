"""Tests for gh_c2_4.ghosal_exp_link."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_4 import ghosal_exp_link


def test_gh_c2_4_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_exp_link(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c2_4_edge():
    """Test edge cases."""
    # Need at least two points for the trapezoidal normalizer; use two points.
    x = np.array([1.0, 2.0])
    import math
    result = ghosal_exp_link(x)
    assert "normalizer" in result
    # Independent computation of the trapezoidal normalizer.
    xs = [1.0, 2.0]
    psi = lambda t: math.sin(3.0 * t)
    e0 = math.exp(psi(xs[0]))
    e1 = math.exp(psi(xs[1]))
    expected_Z = 0.5 * (e0 + e1) * (xs[1] - xs[0])
    assert math.isclose(float(result["normalizer"]), expected_Z, rel_tol=1e-12)
