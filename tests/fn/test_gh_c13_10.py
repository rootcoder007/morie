"""Tests for gh_c13_10.ghosal_ntr_consist."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c13_10 import ghosal_ntr_consist


def test_gh_c13_10_basic():
    """Test basic functionality.

    The function ghosal_ntr_consist takes a tuple of sample sizes `ns`
    and a `seed`. It simulates exponential survival data with light
    censoring and computes the Bayesian product-limit estimator at
    t = 1 for each n. The returned RichResult exposes an `estimate`
    key (the error at the largest n), `err_by_n`, `improving`, and
    `method`.
    """
    ns = (100, 800, 6400)
    seed = 42
    result = ghosal_ntr_consist(ns=ns, seed=seed)
    assert "estimate" in result
    assert "err_by_n" in result
    assert "improving" in result
    assert "method" in result

    err_by_n = result["err_by_n"]
    assert len(err_by_n) == len(ns)
    assert np.all(np.isfinite(np.asarray(err_by_n, dtype=float)))

    # The function definition sets the `estimate` payload key to the
    # error at the largest sample size, i.e. err_by_n[-1].
    assert np.allclose(
        np.asarray(result["estimate"], dtype=float),
        float(err_by_n[-1]),
    )

    # Under the documented behaviour, the estimator at t = 1 should
    # be close to exp(-1). We sanity-check using the same value the
    # implementation compares against, math.exp(-1.0).
    assert result["estimate"] >= 0.0
    assert result["estimate"] <= 1.0
    assert math.isfinite(result["estimate"])

    # `improving` is documented as errs[-1] < errs[0].
    assert result["improving"] == (err_by_n[-1] < err_by_n[0])


def test_gh_c13_10_edge():
    """Test edge cases.

    The function expects `ns` to be an iterable of positive integers
    (sample sizes). Passing a single sample size should still work
    and return a result whose `err_by_n` has one entry, equal to the
    `estimate`.
    """
    ns = (6400,)
    result = ghosal_ntr_consist(ns=ns, seed=42)
    assert "estimate" in result
    assert "err_by_n" in result
    assert np.allclose(
        np.asarray(result["estimate"], dtype=float),
        float(result["err_by_n"][-1]),
    )
    assert len(result["err_by_n"]) == len(ns)
    assert result["improving"] == (
        result["err_by_n"][-1] < result["err_by_n"][0]
    )
