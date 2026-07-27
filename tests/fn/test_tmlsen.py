"""Tests for tmlsen."""

import numpy as np
import pytest

from morie.fn.tmlsen import tmle_sensitivity_unmeasured


def _confounded(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(W @ np.array([1.0, -0.5, 0.3]))))
    A = (rng.random(n) < e).astype(float)
    y = 2.0 * A + W @ np.full(3, 1.0) + rng.normal(scale=0.5, size=n)
    return y, A, W


def test_tmlsen_basic():
    y, A, W = _confounded()
    out = tmle_sensitivity_unmeasured(y, A, W, gamma_grid=[1.0, 1.5, 3.0])
    widths = out["upper"] - out["lower"]
    assert widths[0] == pytest.approx(0.0, abs=1e-6)
    assert widths[1] < widths[2]


def test_tmlsen_edge():
    y, A, W = _confounded()
    with pytest.raises(ValueError):
        tmle_sensitivity_unmeasured(y, A, W, gamma_grid=[0.5])
