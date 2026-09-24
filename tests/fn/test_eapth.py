"""Tests for eapth.eap_theta_estimator."""

import math

from morie.fn import _array_core as np

from morie.fn.eapth import eap_theta_estimator


def test_eapth_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_items = 10
    y = rng.integers(0, 2, n_items)
    b = rng.normal(0, 1, n_items)
    a = rng.uniform(0.5, 2.0, n_items)
    result = eap_theta_estimator(y, a=a, b=b)
    assert isinstance(result, dict)
    assert "theta" in result
    assert "se" in result
    assert "posterior_sd" in result
    assert "prior_mean" in result
    assert "prior_sd" in result
    assert "n_nodes" in result
    assert "n_items" in result
    assert "method" in result
    assert math.isfinite(result["theta"])
    assert math.isfinite(result["se"])
    assert result["se"] >= 0
    assert result["n_items"] == n_items
    assert result["n_nodes"] == 61


def test_eapth_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n_items = 5
    y = rng.integers(0, 2, n_items)
    b = rng.normal(0, 1, n_items)
    result = eap_theta_estimator(y, b=b, n_nodes=11, prior=(0.0, 2.0))
    assert isinstance(result, dict)
    assert math.isfinite(result["theta"])
    assert math.isfinite(result["se"])
    assert result["n_items"] == n_items
    assert result["n_nodes"] == 11
    assert result["prior_mean"] == 0.0
    assert result["prior_sd"] == 2.0
