"""Tests for gh_c9_7.ghosal_whittle_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_7 import ghosal_whittle_crt


def test_gh_c9_7_basic():
    """Test basic functionality."""
    ns = (256, 1024, 4096)
    n_bins = 6
    seed = 42
    result = ghosal_whittle_crt(ns=ns, n_bins=n_bins, seed=seed)

    # Documented keys in the payload
    assert "estimate" in result
    assert "err_by_n" in result
    assert "improving" in result
    assert "method" in result

    est = result["estimate"]
    err_by_n = result["err_by_n"]
    improving = result["improving"]

    # 'estimate' is the error for the largest n (last entry of err_by_n)
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    assert isinstance(err_by_n, list) and len(err_by_n) == len(ns)
    assert np.all(np.isfinite(np.asarray(err_by_n, dtype=float)))
    assert improving == (err_by_n[-1] < err_by_n[0])

    # Independent expectation: 'estimate' must equal the last entry of err_by_n
    assert est == err_by_n[-1]


def test_gh_c9_7_edge():
    """Test edge cases: single (small) n, no convergence expected."""
    ns = (128,)
    n_bins = 6
    seed = 0
    result = ghosal_whittle_crt(ns=ns, n_bins=n_bins, seed=seed)

    # Same documented keys are always present
    for key in ("estimate", "err_by_n", "improving", "method"):
        assert key in result

    err_by_n = result["err_by_n"]
    assert isinstance(err_by_n, list) and len(err_by_n) == 1

    # With a single n there is nothing to improve upon
    assert result["improving"] is False

    # Independent recomputation of the flat (white-noise) spectrum density
    # that the binned estimators target: 1 / (2*pi).
    # Every binned posterior mean is a convex combination
    #   (0.5 * truth + S_b) / (0.5 + C_b)
    # of truth = 1/(2*pi) and the bin average S_b/C_b, so each bin mean
    # lies in [min(truth, S_b/C_b), max(truth, S_b/C_b)] and therefore
    # its distance to truth is at most the distance of S_b/C_b to truth.
    # Since err_by_n[0] is the mean absolute deviation across bins, it
    # must be >= 0 and finite.
    assert err_by_n[0] >= 0.0
    assert np.isfinite(err_by_n[0])

    # And 'estimate' equals the only element of err_by_n
    assert result["estimate"] == err_by_n[0]
