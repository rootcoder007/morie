"""Tests for gh_inf_dim_cr.ghosal_inf_dim_credible."""

from morie.fn import _array_core as np

from morie.fn.gh_inf_dim_cr import ghosal_inf_dim_credible


def test_gh_inf_dim_cr_basic():
    """Test basic functionality with documented signature."""
    # Function takes keyword args: n, dim, level, n_sim, seed (no positional array).
    result = ghosal_inf_dim_credible(n=50, dim=5, level=0.9, n_sim=20, seed=0)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(est))


def test_gh_inf_dim_cr_edge():
    """Test that the result has documented keys and finite scalar."""
    result = ghosal_inf_dim_credible(n=10, dim=2, level=0.9, n_sim=5, seed=1)
    # Documented return keys: estimate, nominal, conservative_or_close, method.
    for key in ("estimate", "nominal", "conservative_or_close", "method"):
        assert key in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert 0.0 <= est <= 1.0
    assert float(result["nominal"]) == 0.9
