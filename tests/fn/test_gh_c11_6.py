"""Tests for gh_c11_6.ghosal_bm_prior."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_6 import ghosal_bm_prior


def test_gh_c11_6_basic():
    """Test basic functionality."""
    n_grid, n_sim, seed = 200, 400, 42
    result = ghosal_bm_prior(n_grid=n_grid, n_sim=n_sim, seed=seed)
    # Documented keys: estimate, theory_min_st, cov_gap, var_gap, method
    assert "estimate" in result
    assert "theory_min_st" in result
    assert "cov_gap" in result
    assert "var_gap" in result
    assert "method" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray([est])))
    # theory_min_st = (n_grid//4) / n_grid = 1/4
    theory_min_st = (n_grid // 4) / n_grid
    # Estimate of Cov(W(s), W(t)) should be close to min(s,t)=1/4 (with small Monte Carlo error)
    assert abs(est - theory_min_st) < 0.05


def test_gh_c11_6_edge():
    """Test edge cases."""
    n_grid, n_sim, seed = 50, 100, 7
    result = ghosal_bm_prior(n_grid=n_grid, n_sim=n_sim, seed=seed)
    s_idx, t_idx = n_grid // 4, n_grid // 2
    theory_min_st = s_idx / n_grid
    est = float(np.asarray(result["estimate"], dtype=float))
    assert result["theory_min_st"] == theory_min_st
    assert abs(est - theory_min_st) < 0.1
    assert "Brownian motion prior" in result["method"]
