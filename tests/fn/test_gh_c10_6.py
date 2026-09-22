"""Tests for gh_c10_6.ghosal_rnd_series_pr."""

from morie.fn import _array_core as np

from morie.fn.gh_c10_6 import ghosal_rnd_series_pr


def test_gh_c10_6_basic():
    """Test basic functionality."""
    result = ghosal_rnd_series_pr(K_true=4, n=1000, lam=0.5, K_max=15, seed=42)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # The posterior mean is a weighted average of K in [0, K_max] with
    # non-negative weights summing to 1, so the estimate must lie in [0, K_max].
    assert 0.0 <= est <= float(K_max := 15)

    # Independent recomputation of the posterior mean from the function's
    # own posterior (documented in the docstring: "Keys: estimate",
    # plus the returned K_posterior). This avoids hard-coding a numeric
    # value produced by the function under test.
    post = result["K_posterior"]
    s = float(np.sum(post))
    assert s > 0.0
    norm = [float(p) / s for p in post]
    independent_mean = sum(k * p for k, p in enumerate(norm))
    assert abs(est - independent_mean) < 1e-9


def test_gh_c10_6_edge():
    """Test edge cases: deterministic seed yields reproducible estimate."""
    r1 = ghosal_rnd_series_pr(K_true=1, n=100, lam=0.5, K_max=5, seed=7)
    r2 = ghosal_rnd_series_pr(K_true=1, n=100, lam=0.5, K_max=5, seed=7)
    e1 = float(np.asarray(r1["estimate"], dtype=float))
    e2 = float(np.asarray(r2["estimate"], dtype=float))
    assert e1 == e2

    # mode_K is the argmax of the posterior and must index a valid K.
    post = r1["K_posterior"]
    mode_K = int(r1["mode_K"])
    assert 0 <= mode_K <= 5
    assert post[mode_K] == max(post)
