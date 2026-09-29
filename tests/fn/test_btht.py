"""Tests for btht.boot_test_hypothesis."""

from morie.fn import _array_core as np
from morie.fn.btht import boot_test_hypothesis


def test_btht_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_test_hypothesis(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btht_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_test_hypothesis(x)
    assert isinstance(result, dict)


def test_null_resampling_p_value_recomputed():
    """Shift to theta0, resample with the Park-Miller stream, count."""
    import pytest

    x = [0.8, 1.9, 0.3, 1.4, 2.2, 0.6, 1.1]
    n, B, th0 = 7, 49, 0.5
    t = sum(x) / n
    sh = [v - t + th0 for v in x]
    s, M, ge, le = 3, 2147483647, 0, 0
    for _ in range(B):
        samp = []
        for _ in range(n):
            s = (16807 * s) % M
            j = min(int((s - 1.0) / (M - 1.0) * n), n - 1)
            samp.append(sh[j])
        tb = sum(samp) / n
        ge += tb >= t
        le += tb <= t
    p = min(1.0, 2 * min((1 + ge) / (B + 1), (1 + le) / (B + 1)))
    r = boot_test_hypothesis(x, theta0=th0, B=B, seed=3)
    assert r["p"] == pytest.approx(p, rel=1e-15)
    assert r["T_hat"] == pytest.approx(t, rel=1e-15)
