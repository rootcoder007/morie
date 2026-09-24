"""Tests for btpair.boot_pairs_regression."""

import pytest

from morie.fn import _array_core as np
from morie.fn.btpair import boot_pairs_regression


def test_btpair_basic():
    """Basic functionality of pairs bootstrap regression."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    B = 50
    result = boot_pairs_regression(X, y, B=B, seed=123, alpha=0.05)
    # The function returns a RichResult that behaves like a dict
    assert isinstance(result, dict)
    # Check a few documented payload keys
    assert "beta_hat" in result
    assert "se" in result
    assert "beta_b" in result
    # Sample size, dimension and replicates are recorded
    assert result["n"] == n
    assert result["p"] == p
    assert result["B"] == B
    # beta_b should contain B replicates, each of length p
    assert len(result["beta_b"]) == B
    for row in result["beta_b"]:
        assert len(row) == p


def test_btpair_edge():
    """Edge cases: invalid alpha raises ValueError."""
    rng = np.random.default_rng(0)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.normal(0, 1, n)
    # alpha must be strictly between 0 and 1
    with pytest.raises(ValueError):
        boot_pairs_regression(X, y, B=10, seed=1, alpha=0.0)
    with pytest.raises(ValueError):
        boot_pairs_regression(X, y, B=10, seed=1, alpha=1.0)
