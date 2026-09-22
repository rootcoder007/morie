"""Tests for gh_c4_14.ghosal_dp_fs_approx."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_14 import ghosal_dp_fs_approx


def test_gh_c4_14_basic():
    """Test basic functionality.

    E(N_eps + 1) = 2 + M * log(1/eps)  (Prop 4.20), so the expected value of
    the returned estimate equals 1 + M * log(1/eps).
    """
    eps = 1e-3
    alpha = 2.0
    seed = 42
    result = ghosal_dp_fs_approx(eps, alpha, seed=seed)

    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)

    # Documented expected support size: E(N_eps + 1) = 2 + M * log(1/eps).
    expected_support_size = 2.0 + alpha * (-float(np.log(eps)))
    expected_estimate_mean = expected_support_size - 1.0
    assert np.isclose(result["expected_support_size"], expected_support_size)

    # Sanity: the actual draw should be in a reasonable neighbourhood of the
    # closed-form mean (loose bound to avoid flakiness).
    assert 0.5 * expected_estimate_mean <= estimate <= 2.0 * expected_estimate_mean


def test_gh_c4_14_edge():
    """Test edge cases.

    With eps extremely small the loop draws many atoms; with eps large the
    loop terminates after one draw.
    """
    # Large eps -> algorithm should only retain the first atom.
    result = ghosal_dp_fs_approx(1e-1, 1.0)
    assert "estimate" in result
    assert float(result["estimate"]) >= 1.0

    # Small eps, alpha=1 -> expected support size = 2 + 1 * log(1/eps).
    eps = 1e-6
    alpha = 1.0
    result2 = ghosal_dp_fs_approx(eps, alpha, seed=0)
    expected_support_size = 2.0 + alpha * (-float(np.log(eps)))
    assert np.isclose(result2["expected_support_size"], expected_support_size)
