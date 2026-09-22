"""Tests for gh_c2_2.ghosal_gp_prior_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_2 import ghosal_gp_prior_def


def test_gh_c2_2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_gp_prior_def(x)
    assert "estimate" in result
    assert "f" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    f_vals = list(result["f"])
    assert len(f_vals) == 5
    assert all(np.isfinite(float(v)) for v in f_vals)
    # Compute expected estimate independently from the returned f values.
    expected_estimate = sum(float(v) for v in f_vals) / len(f_vals)
    assert float(result["estimate"]) == expected_estimate


def test_gh_c2_2_edge():
    """Test edge cases: single-point input."""
    x_in = np.array([42.0])
    result = ghosal_gp_prior_def(x_in)
    assert "estimate" in result
    assert "f" in result
    f_vals = list(result["f"])
    assert len(f_vals) == 1
    assert np.isfinite(float(result["estimate"]))
    assert np.isfinite(float(f_vals[0]))
    # For a single input the GP draw equals its mean (0) plus the diagonal
    # variance factor times the standard normal; the estimate is the single draw.
    assert float(result["estimate"]) == float(f_vals[0])
