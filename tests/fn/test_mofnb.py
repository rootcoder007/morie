"""Tests for morie.fn.mofnb — m-out-of-n bootstrap."""

import pytest

from morie.fn import _array_core as np
from morie.fn.mofnb import mofnb


def test_default_m():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    result = mofnb(x, n_boot=200, seed=7)
    assert result["m"] == int(1000 ** (2.0 / 3.0))


def test_custom_m():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = mofnb(x, m=20, n_boot=100, seed=1)
    assert result["m"] == 20


def test_m_too_large_raises():
    with pytest.raises(ValueError, match="m must be <= n"):
        mofnb(np.array([1.0, 2.0]), m=10)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mofnb(np.array([]))


def test_m_out_of_n_rescaling_recomputed():
    """Bootstrap of size m, deviations rescaled by sqrt(n/m) around theta_hat."""
    import math

    from morie.fn import _array_core as np

    x = np.array([2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3])
    n, m, B = 9, 4, 30
    th = float(np.mean(x))
    rng = np.random.default_rng(2)
    boot = [float(np.mean(x[rng.choice(n, size=m, replace=True)])) for _ in range(B)]
    sc = [th + (b - th) / math.sqrt(m / n) for b in boot]
    mm = sum(sc) / B
    r = mofnb(x, m=m, n_boot=B, seed=2)
    assert list(r["boot_distribution"]) == pytest.approx(boot, rel=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(sum((v - mm) ** 2 for v in sc) / (B - 1)), rel=1e-12)
