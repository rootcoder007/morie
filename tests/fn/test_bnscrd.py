"""Tests for bnscrd.bound_causal_rd."""

import numpy as np
import pytest

from morie.fn.bnscrd import bound_causal_rd


def test_bnscrd_basic():
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(scale=0.2, size=n)
    obs = rng.random(n) > 0.15
    out = bound_causal_rd(y, x, 0.0, observed=obs, bandwidth=0.5, y_min=-2, y_max=3)
    assert out["lower"] <= out["estimate"] <= out["upper"]
    assert out["n_missing"] > 0


def test_bnscrd_edge():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(scale=0.2, size=n)
    full = bound_causal_rd(y, x, 0.0, bandwidth=0.5)
    assert full["width"] == pytest.approx(0.0, abs=1e-9)  # nothing missing
    with pytest.raises(ValueError):
        bound_causal_rd(y, x, 0.0, bandwidth=-1.0)
