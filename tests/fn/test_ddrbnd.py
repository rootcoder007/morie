"""Tests for ddrbnd.deer_dr_bounds."""

import numpy as np
import pytest

from morie.fn.ddrbnd import deer_dr_bounds


def test_ddrbnd_basic():
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.normal(size=n)
    z = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)
    typ = rng.choice(["c", "a", "n"], size=n, p=[0.6, 0.2, 0.2])
    d = np.where(typ == "a", 1.0, np.where(typ == "n", 0.0, z))
    y = 2.0 * d * (typ == "c") + 0.5 * x + rng.normal(scale=0.5, size=n)
    out = deer_dr_bounds(y, d, z, x)
    assert out["late"] == pytest.approx(2.0, abs=0.3)
    assert out["defier_check"] is True


def test_ddrbnd_edge():
    with pytest.raises(ValueError):
        deer_dr_bounds([1.0, 2.0], [1, 0], [1, 1])  # no instrument variation
    with pytest.raises(ValueError):
        deer_dr_bounds([1.0, 2.0], [0.5, 0.0], [1, 0])  # non-binary D
