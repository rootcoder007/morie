"""Tests for drvst.dr_did_variance_stab."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.drvst import dr_did_variance_stab


def test_drvst_basic():
    """Test basic functionality."""
    n = 40
    p = 3
    rng_y = np.random.default_rng(43)
    rng_D = np.random.default_rng(42)
    rng_X = np.random.default_rng(44)
    y = rng_y.normal(0, 1, n)
    D = rng_D.integers(0, 2, n)
    # Ensure D contains both treated and control units
    if not (0 in D and 1 in D):
        D[0] = 0
        D[1] = 1
    X = rng_X.normal(0, 1, (n, p))
    result = dr_did_variance_stab(y, D, X)
    assert isinstance(result, dict)
    expected_keys = {"estimate", "tau_raw", "ess_raw", "ess_stab", "ess_ratio", "cap", "max_weight", "n"}
    assert expected_keys.issubset(result.keys())
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
    assert result["cap"] >= 0.0


def test_drvst_edge():
    """Test that D without control units raises ValueError."""
    n = 40
    p = 3
    rng_y = np.random.default_rng(45)
    rng_X = np.random.default_rng(46)
    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, p))
    D = [0] * n
    with pytest.raises(ValueError):
        dr_did_variance_stab(y, D, X)
