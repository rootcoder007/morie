"""Tests for volhar."""

import numpy as np
import pytest

from morie.fn.volhar import vol_har_rv


def _sim(seed=42, n=500):
    rng = np.random.default_rng(seed)
    rv = np.empty(n); rv[:22] = 1.0
    for t in range(22, n):
        m = 0.05 + 0.4 * rv[t-1] + 0.3 * rv[t-5:t].mean() + 0.2 * rv[t-22:t].mean()
        rv[t] = max(m + rng.normal(scale=0.05), 1e-4)
    return rv


def test_volhar_basic():
    out = vol_har_rv(_sim(), h=2)
    assert out["coefficients"][1] == pytest.approx(0.4, abs=0.15)
    assert out["forecast"].size == 2


def test_volhar_edge():
    with pytest.raises(ValueError):
        vol_har_rv(np.ones(10))
    with pytest.raises(ValueError):
        vol_har_rv(_sim(), h=0)
