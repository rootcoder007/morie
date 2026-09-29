"""Tests for jkest.jackknife_estimator."""

from morie.fn import _array_core as np
from morie.fn.jkest import jackknife_estimator


def test_jkest_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jackknife_estimator(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_jkest_edge():
    """Test edge cases."""
    result = jackknife_estimator(np.array([42.0]))
    assert result["n"] == 1


def test_jackknife_estimator_recomputed():
    """For the plug-in variance the jackknife correction gives the unbiased
    variance exactly."""
    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0, 5.0]
    n = 6

    def pv(v):
        m = sum(v) / len(v)
        return sum((t - m) ** 2 for t in v) / len(v)

    r = jackknife_estimator(x, statistic=lambda a: pv(list(a)))
    m = sum(x) / n
    assert r["estimate"] == pytest.approx(sum((t - m) ** 2 for t in x) / (n - 1), rel=1e-12)
    loo = [pv(x[:i] + x[i + 1 :]) for i in range(n)]
    tb = sum(loo) / n
    assert r["var"] == pytest.approx((n - 1) / n * sum((t - tb) ** 2 for t in loo), rel=1e-12)
