"""Tests for gh_c9_4.ghosal_dpm_norm_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_4 import ghosal_dpm_norm_crt


def test_gh_c9_4_basic():
    """Test basic functionality."""
    ns = (10, 100, 3000)
    alpha = 1.0
    sd = 0.3
    seed = 42
    result = ghosal_dpm_norm_crt(ns=ns, alpha=alpha, sd=sd, seed=seed)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    # Estimate is the mean of the query-point absolute errors at n = ns[-1],
    # so it must lie in [0, 1] (integrable density error on a bounded interval).
    assert 0.0 <= float(est) <= 1.0
    # The function also returns a full error curve and an improving flag.
    assert "err_by_n" in result
    assert "improving" in result
    assert "method" in result
    err_curve = np.asarray(result["err_by_n"], dtype=float)
    assert err_curve.shape == (len(ns),)


def test_gh_c9_4_edge():
    """Test edge cases."""
    result = ghosal_dpm_norm_crt(ns=(1,), alpha=1.0, sd=0.3, seed=0)
    assert "estimate" in result
    assert "err_by_n" in result
    # With a single sample point the error curve has length 1.
    assert len(result["err_by_n"]) == 1
    # The "improving" flag is a boolean, not an integer count.
    assert isinstance(result["improving"], bool)
