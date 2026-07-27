"""Tests for tmllng."""

import numpy as np
import pytest

from morie.fn.tmllng import tmle_longitudinal
from morie.fn.tmltvc import tmle_time_varying_confound


def _tv(seed=42, n=3000):
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = A1 + A2 + L2 + rng.normal(scale=0.5, size=n)
    return y, np.c_[A1, A2], np.c_[L1, L2]


def test_tmllng_basic():
    y, A, L = _tv()
    out = tmle_longitudinal(y, A, L)
    hi = tmle_time_varying_confound(y, A, L, regime=np.ones(2))
    lo = tmle_time_varying_confound(y, A, L, regime=np.zeros(2))
    assert out["estimate"] == pytest.approx(hi["estimate"] - lo["estimate"])
    assert out["n_periods"] == 2


def test_tmllng_edge():
    y, A, L = _tv(n=500)
    with pytest.raises(ValueError):
        tmle_longitudinal(y[:10], A, L)  # shape mismatch
