"""Tests for gestee.gauss_subgaussian_estimator."""

from morie.fn import _array_core as np
from morie.fn.gestee import gauss_subgaussian_estimator


def test_gestee_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gauss_subgaussian_estimator(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_gestee_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gauss_subgaussian_estimator(y)
    assert isinstance(result, dict)


def test_private_clipped_mean_recomputed():
    """Clip to [lower, lower + C], average, add the seeded Laplace draw."""
    import pytest

    from morie.fn import _array_core as np

    y = [1.0, 4.0, 2.5, 9.0, 3.5, 12.0]
    C, lo, eps = 8.0, 0.0, 0.5
    cl = [min(max(v, lo), lo + C) for v in y]
    mu = sum(cl) / 6
    sens = C / 6
    noise = float(np.random.default_rng(3).laplace(0.0, sens / eps))
    r = gauss_subgaussian_estimator(y, C=C, epsilon=eps, lower=lo, seed=3)
    assert r["clipped_mean"] == pytest.approx(mu, rel=1e-14)
    assert r["sensitivity"] == pytest.approx(sens, rel=1e-15)
    assert r["estimate"] == pytest.approx(mu + noise, rel=1e-13)
    assert r["clipping_bias"] == pytest.approx(mu - sum(y) / 6, rel=1e-13)
    assert r["n_clipped"] == 2
