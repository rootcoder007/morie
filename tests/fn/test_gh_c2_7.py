"""Tests for gh_c2_7.ghosal_bernstein_feller."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_7 import ghosal_bernstein_feller


def test_gh_c2_7_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bernstein_feller(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c2_7_edge():
    """Test edge cases."""
    x = np.array([42.0])
    K = 30
    result = ghosal_bernstein_feller(x, K=K)
    assert "F_K" in result
    fk = result["F_K"]
    assert len(fk) == 1
    # Bernstein approximation of F(t) = t^2 at t = 1.0 must equal 1.0
    assert abs(float(np.asarray(fk[0], dtype=float)) - 1.0) < 1e-12
    assert "sup_error" in result
    # With F(t)=t^2 and t=1, the sup error is 0
    assert abs(float(np.asarray(result["sup_error"], dtype=float)) - 0.0) < 1e-12
