"""Tests for tmlpoy."""

import numpy as np
import pytest

from morie.fn.tmlpoy import tmle_propensity_only


def _confounded(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(W @ np.array([1.0, -0.5, 0.3]))))
    A = (rng.random(n) < e).astype(float)
    y = 2.0 * A + W @ np.full(3, 1.0) + rng.normal(scale=0.5, size=n)
    return y, A, W


def test_tmlpoy_basic():
    y, A, W = _confounded()
    out = tmle_propensity_only(y, A, W)
    assert out["ate"] == pytest.approx(2.0, abs=0.3)  # null Q, correct g
    assert out["ate_full"] == pytest.approx(2.0, abs=0.3)


def test_tmlpoy_edge():
    y, A, W = _confounded()
    with pytest.raises(ValueError):
        tmle_propensity_only(y, np.zeros_like(A), W)  # one arm
    with pytest.raises(ValueError):
        tmle_propensity_only(y, A, W, trunc=0.9)
