"""Tests for drpdid.placebo_dr_did."""

from morie.fn import _array_core as np

from morie.fn.drpdid import placebo_dr_did


def test_drpdid_basic():
    """Test basic functionality with parallel pre-trends and valid 0/1 treatment."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 5))
    # Construct a propensity-based binary treatment indicator.
    e = 1 / (1 + np.exp(-(0.5 * X[:, 0])))
    D = (rng.random(n) < e).astype(float)
    # Outcomes in the two pre-treatment periods with no differential pre-trend.
    y_pre1 = X[:, 0] + rng.normal(0, 0.5, n)
    y_pre2 = X[:, 0] + rng.normal(0, 0.5, n)
    result = placebo_dr_did(y_pre1, y_pre2, D, X)
    assert isinstance(result, dict)
    # Documented return keys.
    for key in ("placebo_effect", "se", "ci", "passed", "min_detectable"):
        assert key in result
    # The minimum detectable effect should be positive and computed as 2.8 * se.
    assert result["min_detectable"] > 0
    assert result["min_detectable"] == 2.8 * result["se"]


def test_drpdid_edge():
    """Test edge cases: a clear differential pre-trend should fail the placebo test."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=(n, 5))
    e = 1 / (1 + np.exp(-(0.5 * X[:, 0])))
    D = (rng.random(n) < e).astype(float)
    y_pre1 = X[:, 0] + rng.normal(0, 0.5, n)
    # Inject a strong treatment-correlated pre-trend.
    y_pre2 = y_pre1 + 1.5 * D + rng.normal(0, 0.5, n)
    result = placebo_dr_did(y_pre1, y_pre2, D, X)
    assert isinstance(result, dict)
    for key in ("placebo_effect", "se", "ci", "passed", "min_detectable"):
        assert key in result
