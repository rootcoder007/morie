"""Tests for gh_besov_prior.ghosal_besov_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_besov_prior import ghosal_besov_prior


def test_gh_besov_prior_basic():
    """Test basic functionality with the documented scalar signature."""
    s = 1.0
    J = 8
    pi_j = 0.5
    seed = 42
    result = ghosal_besov_prior(s=s, J=J, pi_j=pi_j, seed=seed)

    # The documented return is a RichResult with an "estimate" key.
    assert "estimate" in result
    assert "n_active" in result
    assert "finite" in result
    assert "method" in result

    est = float(np.asarray(result["estimate"], dtype=float).reshape(-1)[0])
    assert np.isfinite(est)

    # Independent recomputation of the Besov-type norm from the formula
    #   estimate = sum_{j=0..J-1} 2^{j(2(s-0.25)+1)} * lvl_j
    # where lvl_j = sum_k theta_{jk}^2 over the active coefficients.
    rng = np.random.default_rng(seed)
    expected_norm = 0.0
    for j in range(J):
        sd = 2.0 ** (-j * (2.0 * s + 1.0) / 2.0)
        lvl = 0.0
        for k in range(2 ** j):
            if float(rng.uniform(0, 1)) < pi_j:
                th = sd * float(rng.normal(0, 1))
                lvl += th * th
        expected_norm += 2.0 ** (j * (2.0 * (s - 0.25) + 1.0)) * lvl

    assert abs(est - expected_norm) < 1e-12


def test_gh_besov_prior_edge():
    """Test edge cases with deterministic, single-level parameters."""
    result = ghosal_besov_prior(s=0.5, J=1, pi_j=0.0, seed=0)

    # With pi_j = 0 every coefficient is the delta_0 atom, so the
    # Besov-norm estimate must be exactly zero.
    est = float(np.asarray(result["estimate"], dtype=float).reshape(-1)[0])
    assert est == 0.0

    # No coefficients are ever activated under pi_j = 0.
    assert int(np.asarray(result["n_active"], dtype=int).reshape(-1)[0]) == 0
    assert bool(result["finite"]) is True
