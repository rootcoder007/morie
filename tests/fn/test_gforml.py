"""Tests for gforml.robins_g_formula."""

import numpy as np
import pytest

from morie.fn.gforml import robins_g_formula


def _tv_dgp(seed, n=3000):
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = A1 + A2 + L2 + rng.normal(scale=0.5, size=n)
    return y, np.c_[A1, A2], np.c_[L1, L2]


def test_gforml_basic():
    # truth: E[Y(1,1)] - E[Y(0,0)] = 1 + 1 + 0.7 = 2.7
    y, A, L = _tv_dgp(42)
    hi = robins_g_formula(y, A, L, [1, 1], n_mc=4000, seed=0)
    lo = robins_g_formula(y, A, L, [0, 0], n_mc=4000, seed=0)
    assert hi["estimate"] - lo["estimate"] == pytest.approx(2.7, abs=0.25)


def test_gforml_edge():
    with pytest.raises(ValueError):
        robins_g_formula([1.0, 2.0], [[0.5, 1]], [[0.0, 0.0]], [1, 1])  # non-binary
    with pytest.raises(ValueError):
        robins_g_formula([1.0], [[1, 0]], [[0.0]], [1, 1])  # shape mismatch
