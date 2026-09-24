"""Tests for ipferd.ipw_with_replicate."""

import math

from morie.fn import _array_core as np

from morie.fn.ipferd import ipw_with_replicate


def test_ipferd_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_w = np.random.default_rng(45)
    rng_R = np.random.default_rng(44)

    n = 100
    B = 10

    y = rng_y.normal(0, 1, n)
    D = rng_D.integers(0, 2, n)
    w = rng_w.uniform(0.5, 2.0, n)
    replicate_weights = rng_R.uniform(0.5, 2.0, (n, B))

    result = ipw_with_replicate(y, D, w, replicate_weights)

    payload = result.payload
    assert "estimate" in payload
    assert "se" in payload
    assert "variance" in payload
    assert "replicate_estimates" in payload
    assert payload["n"] == n
    assert payload["n_replicates"] == B
    assert math.isfinite(payload["estimate"])
    assert math.isfinite(payload["se"])
    assert payload["se"] >= 0
    assert payload["variance"] >= 0
    assert len(payload["replicate_estimates"]) == B


def test_ipferd_edge():
    """Test edge cases with custom scale."""
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_w = np.random.default_rng(45)
    rng_R = np.random.default_rng(44)

    n = 40
    B = 5

    y = rng_y.normal(0, 1, n)
    D = rng_D.integers(0, 2, n)
    w = rng_w.uniform(0.1, 3.0, n)
    replicate_weights = rng_R.uniform(0.5, 2.0, (n, B))

    result = ipw_with_replicate(y, D, w, replicate_weights, scale=0.5)

    payload = result.payload
    assert "estimate" in payload
    assert "se" in payload
    assert payload["n"] == n
    assert payload["n_replicates"] == B
    assert payload["scale"] == 0.5
    assert math.isfinite(payload["estimate"])
    assert math.isfinite(payload["se"])
    assert payload["se"] >= 0
