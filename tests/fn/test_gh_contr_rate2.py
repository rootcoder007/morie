"""Tests for gh_contr_rate2.ghosal_contraction_rate_iid."""

from morie.fn import _array_core as np

from morie.fn.gh_contr_rate2 import ghosal_contraction_rate_iid


def test_gh_contr_rate2_basic():
    """Test basic functionality with documented defaults."""
    p0 = (0.4, 0.3, 0.2, 0.1)
    ns = (50, 500, 5000)
    result = ghosal_contraction_rate_iid(p0=p0, ns=ns, seed=42)
    # Documented keys
    assert "estimate" in result
    assert "l1_by_n" in result
    assert "half_rate" in result
    assert "method" in result
    # estimate must be finite and l1_by_n must have one entry per n
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(est)))
    l1 = np.asarray(result["l1_by_n"], dtype=float)
    assert l1.shape == (len(ns),)
    # All l1 distances must be positive (posterior expected L1 distance > 0)
    assert np.all(l1 > 0)
    # All entries in p0 normalized region and sum to 1; ns strictly increasing
    assert len(p0) >= 2
    assert all(ns[i] < ns[i + 1] for i in range(len(ns) - 1))
    # Independent computation of contraction rate via plain arithmetic
    p0_sum = sum(p0)
    p0n = [pk / p0_sum for pk in p0]
    # With seed=42 we cannot reproduce dists without re-running RNG;
    # instead verify the rate definition holds using returned l1 values.
    rate_check = np.log(l1[0] / l1[-1]) / np.log(float(ns[-1]) / ns[0])
    assert abs(float(rate_check) - est) < 1e-9


def test_gh_contr_rate2_edge():
    """Test edge cases: equal weights, minimal length-2 support."""
    p0 = (0.5, 0.5)
    ns = (50, 500, 5000)
    result = ghosal_contraction_rate_iid(p0=p0, ns=ns, seed=42)
    assert "estimate" in result
    assert "l1_by_n" in result
    l1 = np.asarray(result["l1_by_n"], dtype=float)
    assert l1.shape == (3,)
    # No division-by-zero: first and last l1 distances are strictly positive
    assert l1[0] > 0
    assert l1[-1] > 0
    assert np.isfinite(float(result["estimate"]))
