"""Tests for gh_c13_11.ghosal_ntr_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_11 import ghosal_ntr_bvm


def test_gh_c13_11_basic():
    """Test basic functionality."""
    n = 1500
    n_sim = 300
    seed = 42
    result = ghosal_ntr_bvm(n=n, n_sim=n_sim, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Additional assertions consistent with the documented BvM behaviour:
    # The asymptotic variance of sqrt(n)*(F_post(1) - F0(1)) should be
    # F0(1)*(1 - F0(1)) for the uncensored exponential truth.
    import math
    F0_1 = 1.0 - math.exp(-1.0)
    expected_var = F0_1 * (1.0 - F0_1)
    assert "efficient_variance" in result
    assert "gap" in result
    # Documented return keys present and finite.
    assert np.all(np.isfinite(np.asarray(result["efficient_variance"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["gap"], dtype=float)))
    # efficient_variance must match the independent closed-form computation.
    assert abs(float(result["efficient_variance"]) - expected_var) < 1e-12
    # gap is the absolute difference between estimate and efficient_variance.
    assert abs(
        float(result["gap"]) - abs(float(result["estimate"]) - expected_var)
    ) < 1e-12


def test_gh_c13_11_edge():
    """Test edge cases with minimum sample sizes."""
    n = 50
    n_sim = 20
    seed = 0
    result = ghosal_ntr_bvm(n=n, n_sim=n_sim, seed=seed)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "efficient_variance" in result
    assert "gap" in result
    # "n" is not a documented return key, so it must not be in the result.
    assert "n" not in result
