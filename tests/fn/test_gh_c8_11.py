"""Tests for gh_c8_11.ghosal_ts_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_11 import ghosal_ts_crt


def test_gh_c8_11_basic():
    """Test basic functionality: spectral-density contraction returns L1 error per n."""
    ns = (256, 2048)
    n_bins = 8
    truth = 1.0 / (2.0 * np.pi) if hasattr(np, "pi") else 1.0 / (2.0 * 3.141592653589793)

    result = ghosal_ts_crt(ns=ns, n_bins=n_bins, seed=42)

    # Documented keys
    assert "estimate" in result
    assert "error_by_n" in result
    assert "contracting" in result
    assert "method" in result

    # Shapes: one estimate per requested sample size
    errs = list(result["error_by_n"])
    assert len(errs) == len(ns)

    # Estimate is the error at the largest n
    assert result["estimate"] == errs[-1]

    # Each entry is a non-negative finite scalar
    for e in errs:
        e_val = float(e)
        assert np.isfinite(e_val)  # N6: was a generator-guessed value
        assert e_val >= 0.0

    # Theoretical upper bound per bin for an exponential with mean truth
    # is its mean; the binned L1 estimator error is therefore at most
    # 2 * truth (since |s/(1+c) - truth| <= s + truth for c >= 1).
    for e in errs:
        assert float(e) <= 2.0 * truth + 1e-12

    # Documented contraction claim: last error should be smaller than first
    assert result["contracting"] == (errs[-1] < errs[0])


def test_gh_c8_11_edge():
    """Test edge cases: single sample size still works and keys are present."""
    result = ghosal_ts_crt(ns=(1024,), n_bins=8, seed=7)

    assert "estimate" in result
    assert "error_by_n" in result
    errs = list(result["error_by_n"])
    assert len(errs) == 1
    assert float(result["estimate"]) == float(errs[0])
    assert np.isfinite(float(result["estimate"]))
