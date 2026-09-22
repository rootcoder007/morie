"""Tests for drovw.dr_overlap_weighted."""

from morie.fn import _array_core as np

from morie.fn.drovw import dr_overlap_weighted


def test_drovw_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    n = 100
    X = rng_X.normal(0, 1, (n, 5))
    e = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.2 * X[:, 1])))
    D = (rng_y.random(n) < e).astype(float)
    y = 2.0 * D + X[:, 0] + 0.5 * rng_y.normal(0, 1, n)
    result = dr_overlap_weighted(y, D, X)
    # Result should be dict-like with the documented return keys.
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci", "estimand", "influence", "max_weight_share"):
        assert key in result
    # The estimand is the ATO (overlap-weighted), as documented.
    assert str(result["estimand"]) == "ATO (overlap-weighted)"
    # ci must be a (lo, hi) 2-tuple.
    ci = result["ci"]
    assert len(ci) == 2
    assert ci[0] < ci[1]
    # influence is per-observation.
    assert len(result["influence"]) == n
    # max_weight_share must be a positive fraction <= 1.
    assert 0.0 < result["max_weight_share"] <= 1.0


def test_drovw_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    n = 100
    X = rng_X.normal(0, 1, (n, 5))
    e = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.2 * X[:, 1])))
    D = (rng_y.random(n) < e).astype(float)
    y = 2.0 * D + X[:, 0] + 0.5 * rng_y.normal(0, 1, n)
    result = dr_overlap_weighted(y, D, X)
    assert isinstance(result, dict)
    # All documented keys must be present.
    for key in ("ate", "se", "ci", "estimand", "influence", "max_weight_share"):
        assert key in result
