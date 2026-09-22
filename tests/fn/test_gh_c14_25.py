"""Tests for gh_c14_25.ghosal_ibp_poisson."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_25 import ghosal_ibp_poisson


def test_gh_c14_25_basic():
    """Test basic functionality against the documented Poisson(alpha H_n) total."""
    n_customers = 25
    alpha = 3.0
    n_sim = 300
    seed = 42

    result = ghosal_ibp_poisson(n_customers=n_customers, alpha=alpha,
                                n_sim=n_sim, seed=seed)

    assert "estimate" in result
    assert "theory" in result
    assert "gap" in result

    estimate = float(np.asarray(result["estimate"], dtype=float))
    theory = float(np.asarray(result["theory"], dtype=float))
    gap = float(np.asarray(result["gap"], dtype=float))

    # Independent computation of H_n and the theoretical mean
    Hn = sum(1.0 / i for i in range(1, n_customers + 1))
    expected_theory = alpha * Hn
    expected_gap = abs(estimate - expected_theory)

    assert np.all(np.isfinite(np.asarray(estimate)))
    assert np.all(np.isfinite(np.asarray(theory)))
    assert abs(theory - expected_theory) < 1e-12
    assert abs(gap - expected_gap) < 1e-12
    # Monte Carlo mean should be within a generous tolerance of the
    # theoretical mean alpha * H_n for 300 simulations.
    assert abs(estimate - expected_theory) < 5.0


def test_gh_c14_25_edge():
    """Test edge case: single customer, single simulation."""
    n_customers = 1
    alpha = 2.0
    n_sim = 1
    seed = 0

    result = ghosal_ibp_poisson(n_customers=n_customers, alpha=alpha,
                                n_sim=n_sim, seed=seed)

    assert "estimate" in result
    assert "theory" in result

    # With n_customers=1, H_1 = 1, so the theoretical total is alpha*1 = alpha.
    Hn = sum(1.0 / i for i in range(1, n_customers + 1))
    expected_theory = alpha * Hn
    theory = float(np.asarray(result["theory"], dtype=float))
    assert abs(theory - expected_theory) < 1e-12
