"""Tests for gh_c12_10.ghosal_wn_lin_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_10 import ghosal_wn_lin_bvm


def test_gh_c12_10_basic():
    """Test basic functionality."""
    result = ghosal_wn_lin_bvm()
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Documented keys
    assert "norm2_L" in result
    assert "gap" in result
    assert "method" in result
    # L = (0.6, 0.8) per docstring -> ||L||^2 = 0.36 + 0.64 = 1.0
    assert abs(result["norm2_L"] - (0.6 ** 2 + 0.8 ** 2)) < 1e-12
    # gap is the absolute difference between sample variance and ||L||^2
    assert abs(result["gap"] - abs(result["estimate"] - result["norm2_L"])) < 1e-12


def test_gh_c12_10_edge():
    """Test with custom n and verify BvM variance ~ ||L||^2."""
    L_coefs = (0.6, 0.8)
    n = 500
    result = ghosal_wn_lin_bvm(L_coefs=L_coefs, n=n, n_sim=500, seed=42)
    assert "estimate" in result
    # Var(sqrt(n) * (L(post) - L(theta0))) should concentrate near ||L||^2
    # Tolerance scales with the magnitude of ||L||^2.
    expected_norm2 = sum(v * v for v in L_coefs)
    assert abs(result["estimate"] - expected_norm2) < 0.2 * expected_norm2
