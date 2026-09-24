"""Tests for hjkest.hajek_estimator."""

from morie.fn import _array_core as np

from morie.fn.hjkest import hajek_estimator


def test_hjkest_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_pi = np.random.default_rng(42)
    n = 100
    y = rng_y.normal(0, 1, n)
    pi = rng_pi.uniform(0.05, 1.0, n)
    result = hajek_estimator(y, pi)
    assert isinstance(result, dict)
    for key in ("mean", "ht_mean_if_N_known", "weight_sum",
                "design_unbiased", "bias_order", "n", "method"):
        assert key in result
    assert result["design_unbiased"] is False
    assert result["n"] == n
    import math
    assert math.isfinite(result["mean"])
    assert math.isfinite(result["weight_sum"])
    assert result["weight_sum"] > 0


def test_hjkest_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_pi = np.random.default_rng(42)
    n = 40
    y = rng_y.normal(0, 1, n)
    pi = rng_pi.uniform(0.1, 0.9, n)
    result = hajek_estimator(y, pi)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert result["method"].startswith("Hajek")
    import math
    assert math.isfinite(result["mean"])
    assert result["design_unbiased"] is False
