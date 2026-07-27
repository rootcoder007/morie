"""Tests for tdmed.two_dimensional_mediation."""

import numpy as np
import pytest

from morie.fn.tdmed import two_dimensional_mediation


def test_tdmed_basic():
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.normal(size=n)
    m1 = 0.7 * x + rng.normal(scale=0.6, size=n)
    m2 = -0.4 * x + rng.normal(scale=0.6, size=n)
    y = 0.2 * x + 1.0 * m1 + 0.5 * m2 + rng.normal(scale=0.6, size=n)
    out = two_dimensional_mediation(x, m1, m2, y)
    assert out["indirect_m1"] == pytest.approx(0.7, abs=0.06)
    assert out["indirect_m2"] == pytest.approx(-0.2, abs=0.06)
    assert out["contrast"] == pytest.approx(out["indirect_m1"] - out["indirect_m2"])


def test_tdmed_edge():
    with pytest.raises(ValueError):
        two_dimensional_mediation([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5)
