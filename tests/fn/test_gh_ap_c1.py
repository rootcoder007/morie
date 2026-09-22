"""Tests for gh_ap_c1.ghosal_covering_num."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_c1 import ghosal_covering_num


def test_gh_ap_c1_basic():
    """Test basic functionality with documented defaults."""
    result = ghosal_covering_num()
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value

    # Independent computation of the literature formula with defaults
    R, eps, dim = 1.0, 0.25, 2
    expected_lo = (R / eps) ** dim
    expected_hi = (3.0 * R / eps) ** dim
    expected_estimate = expected_hi

    assert np.isclose(estimate, expected_estimate)
    assert np.isclose(float(np.asarray(result["lower"], dtype=float)), expected_lo)
    assert np.isclose(float(np.asarray(result["upper"], dtype=float)), expected_hi)
    assert np.isclose(float(np.asarray(result["log_upper"], dtype=float)),
                      dim * np.log(3.0 * R / eps))


def test_gh_ap_c1_custom():
    """Test with explicit documented arguments."""
    R, eps, dim = 2.0, 0.5, 3
    result = ghosal_covering_num(radius_set=R, eps=eps, dim=dim)

    expected_lo = (R / eps) ** dim
    expected_hi = (3.0 * R / eps) ** dim

    assert np.isclose(float(np.asarray(result["estimate"], dtype=float)), expected_hi)
    assert np.isclose(float(np.asarray(result["lower"], dtype=float)), expected_lo)
    assert np.isclose(float(np.asarray(result["upper"], dtype=float)), expected_hi)
    assert np.isclose(float(np.asarray(result["log_upper"], dtype=float)),
                      dim * np.log(3.0 * R / eps))


def test_gh_ap_c1_edge():
    """Test edge case: radius_set equals eps, dim=1."""
    R, eps, dim = 1.0, 1.0, 1
    result = ghosal_covering_num(radius_set=R, eps=eps, dim=dim)

    expected_lo = (R / eps) ** dim  # 1.0
    expected_hi = (3.0 * R / eps) ** dim  # 3.0

    assert np.isclose(float(np.asarray(result["lower"], dtype=float)), expected_lo)
    assert np.isclose(float(np.asarray(result["upper"], dtype=float)), expected_hi)
    assert np.isclose(float(np.asarray(result["estimate"], dtype=float)), expected_hi)
    # Bracket invariant from the docstring: lower <= estimate <= upper
    lo = float(np.asarray(result["lower"], dtype=float))
    hi = float(np.asarray(result["upper"], dtype=float))
    est = float(np.asarray(result["estimate"], dtype=float))
    assert lo <= est <= hi
