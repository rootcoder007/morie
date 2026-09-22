"""Tests for gh_c8_1.ghosal_crt_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_1 import ghosal_crt_def


def test_gh_c8_1_basic():
    """Test basic functionality."""
    result = ghosal_crt_def(theta0=0.4, n=400, M_list=(1.0, 3.0, 9.0), seed=42)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    assert float(est) >= 0.0
    # Documented return keys
    for key in ("estimate", "mass_bound_by_M", "eps_n", "decreasing_in_M", "method"):
        assert key in result
    # eps_n = n^{-1/2} for Bernoulli (Ex 8.3)
    assert result["eps_n"] == 1.0 / np.sqrt(400)
    # mass_bound_by_M is non-increasing in M (Chebyshev bound tightens)
    masses = list(result["mass_bound_by_M"])
    assert len(masses) == 3
    assert all(0.0 <= m <= 1.0 for m in masses)
    assert result["decreasing_in_M"] is True


def test_gh_c8_1_edge():
    """Test edge cases with n=1."""
    result = ghosal_crt_def(theta0=0.5, n=1, M_list=(1.0, 3.0, 9.0), seed=0)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    # eps_n = n^{-1/2}
    assert result["eps_n"] == 1.0 / np.sqrt(1)
