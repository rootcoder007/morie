"""Tests for ggrcst.granger_causality."""

import numpy as np
import pytest

from morie.fn.ggrcst import granger_causality


def test_ggrcst_basic():
    rng = np.random.default_rng(42)
    n = 1500
    x = np.zeros(n); y = np.zeros(n)
    ex = rng.normal(size=n); ey = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + ey[t]
    fwd = granger_causality(x, y, p=1)
    rev = granger_causality(y, x, p=1)
    assert fwd["p_value"] < 0.01
    assert rev["p_value"] > 0.01  # measured ~0.6


def test_ggrcst_edge():
    with pytest.raises(ValueError):
        granger_causality([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], p=0)
    with pytest.raises(ValueError):
        granger_causality([1.0, 2.0], [1.0, 2.0], p=1)  # too short
