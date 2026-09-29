"""Tests for pposm.posterior_predictive_mean."""

from morie.fn import _array_core as np
from morie.fn.pposm import posterior_predictive_mean


def test_pposm_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_mean(samples)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_pposm_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    result = posterior_predictive_mean(samples)
    assert isinstance(result, dict)


def test_predictive_summaries_recomputed():
    import math

    import pytest

    Y = [[1.0, 2.0, 3.0], [2.0, 2.5, 4.0], [0.5, 1.5, 2.5], [3.0, 3.5, 5.0]]
    rm = [sum(r) / 3 for r in Y]
    pooled = [v for r in Y for v in r]
    mp = sum(pooled) / 12
    r = posterior_predictive_mean(Y)
    assert r["estimate"] == pytest.approx(sum(rm) / 4, rel=1e-14)
    assert r["sd_pooled"] == pytest.approx(math.sqrt(sum((v - mp) ** 2 for v in pooled) / 11), rel=1e-13)
    q = sorted(pooled)
    assert (r["ci_lower"], r["ci_upper"]) == (q[math.floor(0.025 * 11)], q[math.ceil(0.975 * 11)])
