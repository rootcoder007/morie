"""Tests for rhomed.rho_critical_mediation."""

import numpy as np
import pytest

from morie.fn.rhomed import rho_critical_mediation
from morie.fn.sensIM import imai_sensitivity_rho


def _simple(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_rhomed_basic():
    x, m, y = _simple()
    out = rho_critical_mediation(x, m, y)
    assert out["rho_critical"] == pytest.approx(imai_sensitivity_rho(x, m, y)["rho_tilde"])
    assert out["abs_rho_critical"] == pytest.approx(abs(out["rho_critical"]))


def test_rhomed_edge():
    with pytest.raises(ValueError):
        rho_critical_mediation([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])  # too few
