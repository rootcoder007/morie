"""Tests for evhpvr.evt_heffernan_tawn."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.evhpvr import evt_heffernan_tawn


def test_evhpvr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    # Generate conditioning variable from an exponential distribution (mean 1)
    uniform = rng.uniform(0, 1, n)
    cond = [-math.log(u) for u in uniform]
    # Generate response variable
    resp = rng.normal(0, 1, n)
    # Build X as n x 2 matrix (list of lists)
    X = [[cond[i], resp[i]] for i in range(n)]
    u = 0.5
    result = evt_heffernan_tawn(X, u)
    assert isinstance(result, dict)
    expected_keys = {"a", "b", "mu_z", "sigma_z", "estimate", "n_exceed", "n"}
    assert expected_keys.issubset(result.keys())
    # n should equal the number of rows
    assert result["n"] == n
    # Compute expected number of exceedances
    expected_n_exceed = sum(1 for c in cond if c > u)
    assert result["n_exceed"] == expected_n_exceed
    assert isinstance(result["n_exceed"], int)
    assert 0 <= result["n_exceed"] <= n
    # estimate equals a
    assert result["estimate"] == result["a"]
    # a should be within [-1, 1]
    assert -1.0 <= result["a"] <= 1.0
    # b should be within [0, 1)
    assert 0.0 <= result["b"] < 1.0
    # mu_z and sigma_z are finite numbers
    assert math.isfinite(result["mu_z"])
    assert math.isfinite(result["sigma_z"])
    assert result["sigma_z"] > 0


def test_evhpvr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    # Generate conditioning variable (exponential)
    uniform = rng.uniform(0, 1, n)
    cond = [-math.log(u) for u in uniform]
    # Generate response variable
    resp = rng.normal(0, 1, n)
    # Invalid: X has three columns instead of two
    X = [[cond[i], resp[i], 0.0] for i in range(n)]
    u = 0.5
    with pytest.raises(ValueError):
        evt_heffernan_tawn(X, u)
