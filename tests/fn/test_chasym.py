"""Tests for chasym.check_asymptote_msm."""

from morie.fn import _array_core as np
from morie.fn.chasym import check_asymptote_msm


def test_chasym_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = check_asymptote_msm(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_chasym_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = check_asymptote_msm(y)
    assert isinstance(result, dict)


def test_sandwich_and_bootstrap_variances_recomputed():
    import pytest

    y = [1.2, 0.4, 2.2, 1.8, 0.9, 1.5, 2.6]
    w = [1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.8]
    n, B = 7, 25
    sw = sum(w)
    th = sum(a * b for a, b in zip(w, y)) / sw
    var_if = sum((a * (b - th)) ** 2 for a, b in zip(w, y)) / sw**2
    s = 11
    reps = []
    for _ in range(B):
        swb = syb = 0.0
        for _ in range(n):
            s = (s * 48271) % 2147483647
            k = min(int((s - 1) / 2147483646.0 * n), n - 1)
            swb += w[k]
            syb += w[k] * y[k]
        reps.append(syb / swb)
    mb = sum(reps) / B
    vb = sum((v - mb) ** 2 for v in reps) / (B - 1)
    r = check_asymptote_msm(y, H=w, B=B, seed=11)
    assert r["theta"] == pytest.approx(th, rel=1e-14)
    assert r["var_if"] == pytest.approx(var_if, rel=1e-13)
    assert r["var_boot"] == pytest.approx(vb, rel=1e-12)
    assert r["ess"] == pytest.approx(sw**2 / sum(v * v for v in w), rel=1e-14)
