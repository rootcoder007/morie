"""Tests for bnsnig.bound_no_unobserved_inv."""

import numpy as np

from morie.fn.bnsnig import bound_no_unobserved_inv


def _make_binary(rng, n):
    return (rng.normal(0, 1, n) > 0).astype(int)


def _make_discrete(rng, n, k):
    return rng.integers(0, k, size=n)


def test_bnsnig_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_other = np.random.default_rng(42)
    n = 100

    y = rng_y.normal(0, 1, n)
    D = _make_binary(rng_other, n)
    X = _make_discrete(rng_other, n, 3)
    X_inv = _make_discrete(rng_other, n, 2)

    result = bound_no_unobserved_inv(y, D, X, X_inv)
    assert isinstance(result, dict)

    # Documented return keys per the docstring.
    expected_keys = {
        "lower", "upper", "width", "estimate",
        "n_strata", "n_cells", "refuted", "n",
    }
    assert expected_keys.issubset(result.keys())

    # Resulting bounds must be well-formed.
    assert result["lower"] <= result["upper"]
    assert result["width"] == result["upper"] - result["lower"]
    assert result["estimate"] == 0.5 * (result["lower"] + result["upper"])
    assert result["n"] == n
    assert result["refuted"] in (0.0, 1.0)
    assert result["n_strata"] >= 1
    assert result["n_cells"] >= result["n_strata"]


def test_bnsnig_edge():
    """Test edge cases with a single X stratum and a single X_inv cell."""
    rng_y = np.random.default_rng(43)
    rng_other = np.random.default_rng(42)
    n = 60

    y = rng_y.normal(0, 1, n)
    D = _make_binary(rng_other, n)
    X = np.zeros(n, dtype=int)
    X_inv = np.zeros(n, dtype=int)

    result = bound_no_unobserved_inv(y, D, X, X_inv)
    assert isinstance(result, dict)
    assert result["n_strata"] == 1
    assert result["n_cells"] == 1
    assert result["n"] == n
    assert result["lower"] <= result["upper"]
