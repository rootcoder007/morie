"""Tests for ksr024 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr024 import kosorok_ch1_partly_linear_logistic


def test_ksr024_basic():
    rng = np.random.default_rng(2)
    n = 600
    Z = rng.standard_normal((n, 1))
    U = rng.random(n)
    p = 1 / (1 + np.exp(-(1.0 * Z[:, 0] + np.sin(3 * U) * 2)))
    out = kosorok_ch1_partly_linear_logistic(rng.binomial(1, p), Z, U, df=6)
    assert out["beta"][0] == pytest.approx(1.0, abs=0.4)


def test_ksr024_edge():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((50, 1))
    with pytest.raises(ValueError):
        kosorok_ch1_partly_linear_logistic(np.full(50, 2.0), Z, rng.random(50))
