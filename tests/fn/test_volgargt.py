"""Tests for volgargt."""

import numpy as np
import pytest

from morie.fn.volgargt import vol_garch_t


def _sim(n=900, seed=0):
    """GARCH(1,1) path: omega 0.05, alpha 0.1, beta 0.85 (uncond. var 1)."""
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    s2 = 1.0
    for t in range(n):
        if t:
            s2 = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * s2
        e[t] = np.sqrt(s2) * rng.standard_normal()
    return e


def test_volgargt_basic():
    out = vol_garch_t(_sim())
    assert np.all(out["sigma2"] > 0)
    assert np.isfinite(out["loglik"])
    assert out["spec"] == "garch"
    assert out["nu"] is not None  # shape estimated, not fixed


def test_volgargt_edge():
    with pytest.raises(ValueError):
        vol_garch_t(np.arange(10.0))  # too short to fit
    with pytest.raises(ValueError):
        vol_garch_t(np.ones(100))  # zero variance
