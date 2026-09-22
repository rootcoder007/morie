"""Tests for gh_ap_g1.ghosal_fin_dir_def."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_ap_g1 import ghosal_fin_dir_def


def test_gh_ap_g1_basic():
    """Test basic functionality with the documented tuple of alpha parameters."""
    alpha = (2.0, 3.0, 5.0)
    result = ghosal_fin_dir_def(alpha)
    assert "estimate" in result
    # Verify keys match the documented payload.
    assert "mean" in result
    assert "log_norm_const" in result
    assert "method" in result
    # Mean must lie on the simplex (sum to 1) with positive entries.
    mean = np.asarray(result["mean"], dtype=float)
    assert np.all(mean > 0.0)
    assert np.all(np.isfinite(mean))
    assert np.allclose(np.sum(mean), 1.0)
    # Estimate must be a finite scalar density evaluated at the mean.
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(estimate)
    assert estimate > 0.0
    # Independent computation of the density at the Dirichlet mean:
    #   p(x) = Gamma(A) / prod Gamma(alpha_j) * prod x_j^(alpha_j - 1)
    # evaluated at x_j = alpha_j / A where A = sum(alpha_j).
    A = float(sum(alpha))
    logC = math.lgamma(A) - sum(math.lgamma(ai) for ai in alpha)
    expected_logdens = logC + sum((ai - 1.0) * math.log(ai / A) for ai in alpha)
    expected = math.exp(expected_logdens)
    assert math.isclose(estimate, expected, rel_tol=1e-12, abs_tol=1e-15)


def test_gh_ap_g1_edge():
    """Test edge case: single-component Dirichlet reduces to a 1-d point mass at x=1."""
    # A Dirichlet on the simplex with one component has mean 1 and density 1.
    result = ghosal_fin_dir_def((42.0,))
    assert "estimate" in result
    assert "mean" in result
    mean = np.asarray(result["mean"], dtype=float)
    assert np.allclose(mean, np.array([1.0]))
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # Independent computation: with one alpha, Gamma(A)/Gamma(alpha) * 1^(alpha-1) = 1.
    alpha = (42.0,)
    A = float(sum(alpha))
    expected_logdens = (math.lgamma(A) - math.lgamma(alpha[0])
                        + (alpha[0] - 1.0) * math.log(alpha[0] / A))
    expected = math.exp(expected_logdens)
    assert math.isclose(estimate, expected, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(estimate, 1.0, rel_tol=1e-12, abs_tol=1e-15)
