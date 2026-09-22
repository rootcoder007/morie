"""Tests for gh_c8_2.ghosal_ggv_thm."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_2 import ghosal_ggv_thm


def test_gh_c8_2_basic():
    """Test basic functionality with all three conditions satisfied."""
    n = 100
    eps_n = 0.1
    C = 1.0
    ne2 = n * eps_n ** 2

    log_prior_mass_B2 = -C * ne2            # boundary: satisfies (i)
    log_entropy = ne2                       # boundary: satisfies (ii)
    log_sieve_tail_mass = -(C + 4.0) * ne2  # boundary: satisfies (iii)

    result = ghosal_ggv_thm(
        n, eps_n, log_prior_mass_B2, log_entropy, log_sieve_tail_mass, C=C
    )

    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # estimate must equal the supplied eps_n when all conditions hold.
    assert result["estimate"] == float(eps_n)
    # n_eps2 = n * eps_n^2, computed independently.
    assert result["n_eps2"] == ne2
    # Conditions are [c_i, c_ii, c_iii]; all three must be True here.
    assert result["conditions"] == [True, True, True]
    assert result["rate_certified"] is True
    assert result["method"] == "basic rate theorem (GvdV 2017 Thm 8.9)"


def test_gh_c8_2_edge():
    """Test that violating a condition yields a non-finite estimate."""
    n = 50
    eps_n = 0.2
    C = 1.0
    ne2 = n * eps_n ** 2

    log_prior_mass_B2 = -(C + 10.0) * ne2  # too small: violates (i)
    log_entropy = ne2
    log_sieve_tail_mass = -(C + 4.0) * ne2

    result = ghosal_ggv_thm(
        n, eps_n, log_prior_mass_B2, log_entropy, log_sieve_tail_mass, C=C
    )

    assert result["estimate"] != result["estimate"]  # NaN check
    assert result["rate_certified"] is False
    assert result["conditions"][0] is False
    assert result["n_eps2"] == ne2
