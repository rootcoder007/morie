"""Tests for gh_sobol_prior.ghosal_sobolev_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_sobol_prior import ghosal_sobolev_prior


def test_gh_sobol_prior_basic():
    """Test basic functionality with a documented scalar smoothness argument."""
    smoothness = 1.0
    n_terms = 2000
    seed = 42
    s = float(smoothness)

    rng = np.random.default_rng(seed)
    th = [float(rng.normal(0, 1)) * float(j) ** (-(s + 0.5))
          for j in range(1, n_terms + 1)]
    expected_estimate = sum(
        float(j) ** (2.0 * (s - 0.25)) * th[j - 1] ** 2
        for j in range(1, n_terms + 1)
    )

    result = ghosal_sobolev_prior(smoothness=smoothness, n_terms=n_terms, seed=seed)

    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert float(result["estimate"]) == expected_estimate


def test_gh_sobol_prior_edge():
    """Test documented return keys (no 'n'; uses scalar smoothness, not an array)."""
    result = ghosal_sobolev_prior(smoothness=0.5, n_terms=10, seed=7)

    # Documented key checks
    assert "estimate" in result
    assert "finite_below_s" in result
    assert "divergent_at_s_partial" in result
    assert "rate" in result
    assert "method" in result

    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert bool(result["finite_below_s"]) is True

    # The documented function does not return an 'n' key
    assert "n" not in result
