"""Tests for plcbo.placebo_refutation."""

import numpy as np
import pytest

from morie.fn.plcbo import placebo_refutation


def _diff(y, tr):
    return float(y[tr == 1].mean() - y[tr == 0].mean())


def test_plcbo_basic():
    rng = np.random.default_rng(42)
    n = 400
    t = (rng.random(n) < 0.5).astype(float)
    y = 2.0 * t + rng.normal(size=n)
    out = placebo_refutation(_diff, y, t, n_simulations=200, seed=0)
    assert out["p_value"] < 0.05
    assert abs(out["placebo_mean"]) < 0.3


def test_plcbo_edge():
    rng = np.random.default_rng(0)
    t = (rng.random(200) < 0.5).astype(float)
    y = rng.normal(size=200)
    assert placebo_refutation(_diff, y, t, n_simulations=100, seed=0)["p_value"] > 0.05
    with pytest.raises(ValueError):
        placebo_refutation("nope", y, t)
