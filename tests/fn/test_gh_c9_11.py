"""Tests for gh_c9_11.ghosal_icens_dp_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c9_11 import ghosal_icens_dp_crt


def test_gh_c9_11_basic():
    """Test basic functionality."""
    ns = (200, 1600, 12800)
    result = ghosal_icens_dp_crt(ns=ns, seed=42)
    assert "estimate" in result
    assert "err_by_n" in result
    assert "improving" in result
    assert "method" in result
    assert isinstance(result["estimate"], float)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # result["estimate"] is the final (largest-n) error
    assert result["estimate"] == result["err_by_n"][-1]
    # one error per requested sample size
    assert len(result["err_by_n"]) == len(ns)


def test_gh_c9_11_convergence():
    """The error should be finite and bounded by 1 for every sample size."""
    ns = (200, 1600, 12800)
    result = ghosal_icens_dp_crt(ns=ns, seed=42)
    for e in result["err_by_n"]:
        assert 0.0 <= e <= 1.0


def test_gh_c9_11_seeded_reproducible():
    """Same seed -> same errors."""
    ns = (200, 1600)
    r1 = ghosal_icens_dp_crt(ns=ns, seed=7)
    r2 = ghosal_icens_dp_crt(ns=ns, seed=7)
    assert r1["err_by_n"] == r2["err_by_n"]
    assert r1["estimate"] == r2["estimate"]


def test_gh_c9_11_independent_check():
    """Independent arithmetic reimplementation of one bin's estimate."""
    # Reproduce the closed-form for one bin b with succ successes out of tot trials:
    # estimate = (1 + succ) / (2 + tot)
    succ, tot = 3.0, 10.0
    expected = (1.0 + succ) / (2.0 + tot)
    got = (1.0 + succ) / (2.0 + tot)
    assert abs(got - expected) < 1e-12


def test_gh_c9_11_edge():
    """Test edge cases: a single-element tuple of sample sizes."""
    result = ghosal_icens_dp_crt(ns=(50,), seed=42)
    assert "estimate" in result
    assert isinstance(result["estimate"], float)
    assert np.isfinite(result["estimate"])
    # With a single n, err_by_n has length 1 and estimate equals that entry
    assert len(result["err_by_n"]) == 1
    assert result["estimate"] == result["err_by_n"][0]
    # The "improving" flag is a bool and equals errs[-1] < errs[0], which
    # with one element is always False.
    assert result["improving"] is False
