"""Tests for causrho.causal_proximal_proxy."""

import numpy as np
import pytest

from morie.fn.causrho import causal_proximal_proxy


def test_causrho_basic():
    rng = np.random.default_rng(42)
    n = 5000
    u = rng.normal(size=n)
    z = u + rng.normal(scale=0.5, size=n)
    w = u + rng.normal(scale=0.5, size=n)
    a = 0.8 * u + rng.normal(scale=0.6, size=n)
    y = 1.0 * a + 1.5 * u + rng.normal(scale=0.5, size=n)
    out = causal_proximal_proxy(y, a, z, w)
    assert abs(out["estimate"] - 1.0) < abs(out["naive"] - 1.0)
    assert out["first_stage_r2"][0] > 0.2


def test_causrho_edge():
    with pytest.raises(ValueError):
        causal_proximal_proxy(np.zeros(10), np.zeros(10), np.zeros((10, 1)), np.zeros((10, 2)))
    with pytest.raises(ValueError):
        causal_proximal_proxy(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))  # too few
