"""Tests for gh_dp_kl_nbhd.ghosal_dp_kl_nbhd_mass."""

from morie.fn import _array_core as np

from morie.fn.gh_dp_kl_nbhd import ghosal_dp_kl_nbhd_mass


def test_gh_dp_kl_nbhd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_kl_nbhd_mass(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_dp_kl_nbhd_edge():
    """Test edge cases."""
    result = ghosal_dp_kl_nbhd_mass(np.array([42.0]))
    # A single-category weight distribution is degenerate: there is exactly one
    # possible Dirichlet sample (all mass on the single category), so the KL
    # divergence between p0 = [1.0] and p = [1.0] is 0, which is below every
    # eps in eps_list, yielding log_mass = log(n_sim/n_sim) = 0.0 for the
    # smallest-eps (last) entry, which is the reported estimate.
    assert result["estimate"] == 0.0
    assert all(lm == 0.0 for lm in result["log_mass_by_eps"])
    assert result["method"] == "KL-neighborhood prior mass (GvdV 2017 sec. 7.2)"
    assert result["monotone"] is True
