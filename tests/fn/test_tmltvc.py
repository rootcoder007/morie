"""Tests for tmltvc."""

from morie.fn import _array_core as np
import pytest

from morie.fn.tmltvc import tmle_time_varying_confound


def _tv(seed=42, n=3000):
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = A1 + A2 + L2 + rng.normal(scale=0.5, size=n)
    return y, np.c_[A1, A2], np.c_[L1, L2]


def test_tmltvc_basic():
    y, A, L = _tv()
    hi = tmle_time_varying_confound(y, A, L, regime=1.0)
    lo = tmle_time_varying_confound(y, A, L, regime=0.0)
    assert hi["estimate"] - lo["estimate"] == pytest.approx(2.7, abs=0.6)
    assert hi["epsilons"].size == 2


def test_tmltvc_edge():
    y, A, L = _tv(n=500)
    with pytest.raises(ValueError):
        tmle_time_varying_confound(y, A * 0.5, L)  # non-binary A
    with pytest.raises(ValueError):
        tmle_time_varying_confound(y, A, L, trunc=0.9)
