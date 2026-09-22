"""Tests for covpop.coverage_correction."""

from morie.fn import _array_core as np

from morie.fn.covpop import coverage_correction


def test_covpop_basic():
    """Test basic functionality with a single stratum (strata=None)."""
    rng_y = np.random.default_rng(43)
    rng_w = np.random.default_rng(45)
    y = rng_y.normal(0, 1, 100)
    weights = rng_w.exponential(1, 100)
    # With strata=None, everything goes into one stratum, so
    # target_totals must have length 1.
    target_totals = [1000.0]
    result = coverage_correction(y, weights, target_totals)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "w_adj" in result
    assert "factors" in result
    assert "total" in result
    assert "n" in result

    # Independent check of the documented formula.
    # One stratum -> factor f = tt / sum(w), w_adj_i = w_i * f,
    # estimate = sum(w_adj_i * y_i) / sum(w_adj_i).
    tt = target_totals[0]
    f = tt / sum(weights)
    w_adj = [weights[i] * f for i in range(len(weights))]
    expected_total = sum(w_adj)
    expected_estimate = sum(w_adj[i] * y[i] for i in range(len(y))) / expected_total
    assert result["factors"] == [f]
    assert result["total"] == expected_total
    assert result["estimate"] == expected_estimate
    assert result["n"] == 100


def test_covpop_edge():
    """Test edge cases with explicit post-strata."""
    rng_y = np.random.default_rng(43)
    rng_w = np.random.default_rng(45)
    rng_s = np.random.default_rng(47)
    n = 100
    y = rng_y.normal(0, 1, n)
    weights = rng_w.exponential(1, n)
    strata = ["a" if i < 50 else "b" for i in range(n)]
    # One target total per stratum, in the order labels first appear.
    target_totals = [1000.0, 2000.0]
    result = coverage_correction(y, weights, target_totals, strata)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
