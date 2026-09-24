"""Tests for mapth.map_theta_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.mapth import map_theta_estimator


def test_mapth_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    m = 40
    y = rng.integers(0, 2, size=m).astype(float)
    a = rng.uniform(0.5, 2.0, size=m)
    b = rng.normal(0.0, 1.0, size=m)
    c = rng.uniform(0.0, 0.3, size=m)
    result = map_theta_estimator(
        y, a=a, b=b, c=c, prior=(0.0, 1.0), bounds=(-6.0, 6.0)
    )
    assert isinstance(result, dict)
    for key in (
        "theta",
        "se",
        "prior_mean",
        "prior_sd",
        "information",
        "posterior_information",
        "shrinkage_vs_ml",
        "exists_for_perfect_patterns",
        "n_items",
        "method",
    ):
        assert key in result
    assert math.isfinite(result["theta"])
    assert -6.0 <= result["theta"] <= 6.0
    assert math.isfinite(result["se"])
    assert result["se"] > 0
    assert result["prior_mean"] == 0.0
    assert result["prior_sd"] == 1.0
    assert result["exists_for_perfect_patterns"] is True
    assert result["n_items"] == m


def test_mapth_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    m = 40
    # All-correct pattern: MLE does not exist here, but MAP should.
    y = np.ones(m)
    b = rng.normal(0.0, 1.0, size=m)
    result = map_theta_estimator(y, b=b, prior=(0.0, 1.0), bounds=(-6.0, 6.0))
    assert isinstance(result, dict)
    assert "theta" in result
    assert math.isfinite(result["theta"])
    assert -6.0 <= result["theta"] <= 6.0
    assert result["exists_for_perfect_patterns"] is True
    assert result["n_items"] == m
    # Prior shrinkage pulls a perfect-score estimate toward the prior mean.
    assert 0.0 <= result["theta"] <= 6.0
