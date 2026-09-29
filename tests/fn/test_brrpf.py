"""Tests for brrpf.brr_prior_posterior."""

from morie.fn import _array_core as np
from morie.fn.brrpf import brr_prior_posterior


def test_brrpf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = brr_prior_posterior(y)
    assert isinstance(result, dict)
    assert "S" in result


def test_brrpf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = brr_prior_posterior(y)
    assert isinstance(result, dict)


def test_brr_default_hyperparameters_recomputed():
    """S = Var(y)(1 - R2)(nu + 2), S_beta = Var(y) R2 (nu_beta + 2)."""
    import pytest

    y = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 4.2]
    m = sum(y) / 7
    v = sum((t - m) ** 2 for t in y) / 6
    r = brr_prior_posterior(y, R2=0.3, nu=4.0, nu_beta=6.0)
    assert r["S"] == pytest.approx(v * 0.7 * 6.0, rel=1e-13)
    assert r["S_beta"] == pytest.approx(v * 0.3 * 8.0, rel=1e-13)
    assert r["var_y"] == pytest.approx(v, rel=1e-13)
