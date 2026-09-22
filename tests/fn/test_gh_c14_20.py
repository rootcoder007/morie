"""Tests for gh_c14_20.ghosal_probit_sbp."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_20 import ghosal_probit_sbp


def test_gh_c14_20_basic():
    """Test basic functionality with a scalar x (documented default shape)."""
    # x is a scalar float per the function signature/default
    x = 0.4
    result = ghosal_probit_sbp(x)

    # The function returns a RichResult with these documented keys
    assert "estimate" in result
    assert "total_mass" in result
    assert "method" in result

    est = np.asarray(result["estimate"], dtype=float)
    # estimate is the first stick-breaking weight w_1 in (0, 1]
    assert np.all(np.isfinite(est))
    assert 0.0 < float(est) <= 1.0

    # total_mass is sum of stick-breaking weights; for a well-behaved
    # construction the partial sums are <= 1, so total_mass <= 1
    tm = float(result["total_mass"])
    assert 0.0 < tm <= 1.0


def test_gh_c14_20_reproducible_seed():
    """Same seed should produce the same estimate."""
    r1 = ghosal_probit_sbp(0.4, n_terms=25, seed=42)
    r2 = ghosal_probit_sbp(0.4, n_terms=25, seed=42)
    assert float(r1["estimate"]) == float(r2["estimate"])


def test_gh_c14_20_independent_formula():
    """Cross-check estimate against an independent computation of the formula."""
    # The formula (sec. 14.9.3) is:
    #   V_k(x) = Phi(mu_k + beta_k * x),  k = 1..n_terms
    #   w_1 = V_1
    #   w_k = V_k * prod_{j<k} (1 - V_j),  k >= 2
    # where Phi(v) = 0.5 * (1 + erf(v / sqrt(2)))
    # and mu_k, beta_k are iid N(0,1) draws (fixed by `seed`).

    import math
    from morie.fn import _array_core as _np  # local alias for the shim's RNG

    x = 0.4
    n_terms = 25
    seed = 42
    rng = _np.random.default_rng(seed)

    def Phi(v):
        return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))

    V = [Phi(float(rng.normal(0, 1))
             + float(rng.normal(0, 1)) * x)
         for _ in range(n_terms)]

    # Independently compute stick-breaking weights
    remaining = 1.0
    weights = []
    for v in V:
        w = remaining * v
        weights.append(w)
        remaining = remaining * (1.0 - v)
    expected_estimate = weights[0]

    result = ghosal_probit_sbp(x, n_terms=n_terms, seed=seed)
    got = float(result["estimate"])
    # Compute tolerance from the independent expression
    assert abs(got - float(expected_estimate)) < 1e-12


def test_gh_c14_20_edge():
    """Test edge case: x at a documented scalar value still yields finite result."""
    # Scalar x, single-element input shape (Python float), no `n` key in output
    result = ghosal_probit_sbp(0.0)
    # The function returns a RichResult whose payload uses these documented keys:
    assert "estimate" in result
    assert "total_mass" in result
    # `n` is not a documented key; assert it's not present rather than
    # equal to 1.
    assert "n" not in result
    # Sanity: estimate is finite
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
