"""Tests for dssoot.bootstrap_indirect."""

from morie.fn import _array_core as np

from morie.fn.dssoot import bootstrap_indirect


def test_dssoot_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    Y = rng.normal(0, 1, 100)
    X = rng.normal(0, 1, 100)
    M = rng.normal(0, 1, 100)
    n_boot = 100
    result = bootstrap_indirect(Y, X, M, n_boot)
    assert isinstance(result, dict)
    for key in ("estimate", "a", "b", "c_prime", "c_total",
                "ci_lo", "ci_hi", "se_boot", "bias", "n_boot", "alpha", "n"):
        assert key in result
    # arity/shape sanity
    assert result["n"] == 100
    assert result["n_boot"] == float(n_boot)
    assert result["alpha"] == 0.05
    # closed-form check for the point estimate ab = a * b
    a = result["a"]
    b = result["b"]
    assert result["estimate"] == a * b


def test_dssoot_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    Y = rng.normal(0, 1, 100)
    X = rng.normal(0, 1, 100)
    M = rng.normal(0, 1, 100)
    n_boot = 100
    result = bootstrap_indirect(Y, X, M, n_boot)
    assert isinstance(result, dict)
    assert "estimate" in result
