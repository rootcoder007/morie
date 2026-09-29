"""Tests for brrvar.brr_variance."""

from morie.fn import _array_core as np
from morie.fn.brrvar import brr_variance


def test_brrvar_basic():
    """Test basic functionality."""
    estimates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(estimates)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_brrvar_edge():
    """Test edge cases."""
    estimates = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_variance(estimates)
    assert isinstance(result, dict)


def test_brr_fay_variance_recomputed():
    import math

    import pytest

    reps = [10.2, 11.8, 9.1, 10.9, 10.4, 9.7]
    r = brr_variance(reps, full_estimate=10.3, fay_k=0.5)
    v = sum((t - 10.3) ** 2 for t in reps) / (6 * 0.25)
    assert r["variance"] == pytest.approx(v, rel=1e-13)
    assert r["se"] == pytest.approx(math.sqrt(v), rel=1e-13)
    assert r["cv"] == pytest.approx(math.sqrt(v) / 10.3, rel=1e-13)
    m = sum(reps) / 6
    assert brr_variance(reps)["variance"] == pytest.approx(sum((t - m) ** 2 for t in reps) / 6, rel=1e-13)
