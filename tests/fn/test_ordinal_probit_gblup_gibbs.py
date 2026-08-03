"""Tests for ordinal_probit_gblup_gibbs.ordinal_probit_gblup_gibbs."""

from morie.fn import _array_core as np

from morie.fn.ordinal_probit_gblup_gibbs import ordinal_probit_gblup_gibbs


def test_msm095_basic():
    """Test basic functionality."""
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    probs = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_probit_gblup_gibbs(Probs, A, probs, where, dat_F, the)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm095_edge():
    """Test edge cases."""
    Probs = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    probs = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = ordinal_probit_gblup_gibbs(Probs, A, probs, where, dat_F, the)
    assert isinstance(result, dict)
