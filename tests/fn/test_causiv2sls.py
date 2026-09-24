"""Tests for causiv2sls.causal_iv_2sls."""

import math

from morie.fn import _array_core as np

from morie.fn.causiv2sls import causal_iv_2sls


def test_causiv2sls_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(41)
    n = 40
    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, 2))
    Z = rng_Z.normal(0, 1, (n, 4))
    result = causal_iv_2sls(y, X, Z)
    assert isinstance(result, dict)
    expected_keys = {
        "beta", "se", "t", "residuals", "fitted",
        "first_stage_F", "order_condition", "overidentified",
        "n_overid_restrictions", "sargan", "sargan_p",
        "vcov_type", "n", "k", "m", "method",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(float(result["first_stage_F"]))
    assert math.isfinite(float(result["sargan"]))
    assert math.isfinite(float(result["sargan_p"]))
    assert int(result["n"]) == n
    assert int(result["k"]) == 3
    assert int(result["m"]) == 5


def test_causiv2sls_edge():
    """Test edge case: clustered sandwich standard errors."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(41)
    n = 40
    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, 2))
    Z = rng_Z.normal(0, 1, (n, 4))
    cluster = [i % 4 for i in range(n)]
    result = causal_iv_2sls(y, X, Z, cluster=cluster)
    assert isinstance(result, dict)
    assert "beta" in result
    assert "se" in result
    assert "first_stage_F" in result
    assert math.isfinite(float(result["first_stage_F"]))
    assert int(result["n"]) == n
