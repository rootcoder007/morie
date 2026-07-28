"""Tests for ksr069 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr069 import kosorok_ch3_cox_likelihood_breslow


def _data(seed=0, n=300, beta=0.8):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 1))
    T = rng.exponential(1.0 / np.exp(Z[:, 0] * beta))
    C = rng.exponential(2.0, n)
    return Z, np.minimum(T, C), (T <= C).astype(float)


def test_ksr069_basic():
    Z, V, d = _data()
    out = kosorok_ch3_cox_likelihood_breslow(0.8, None, Z, V, d)
    # the SE comes from the observed information and must be finite --
    # the previous body took np.std of a scalar and returned NaN
    assert np.isfinite(out["se"])
    assert out["se"] > 0
    assert out["information"][0, 0] > 0
    assert out["estimate"] == pytest.approx(0.8, abs=0.3)
    assert np.all(np.diff(out["breslow_cumhaz"]) >= 0)  # monotone


def test_ksr069_edge():
    Z, V, d = _data()
    with pytest.raises(ValueError):
        kosorok_ch3_cox_likelihood_breslow(0.8, None, Z, V, np.zeros(len(d)))  # no events
    with pytest.raises(ValueError):
        kosorok_ch3_cox_likelihood_breslow([0.8, 0.2], None, Z, V, d)  # beta/Z mismatch
