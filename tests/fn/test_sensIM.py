"""Tests for sensIM.imai_sensitivity_rho."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sensIM import imai_sensitivity_rho


def _simple(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_sensIM_basic():
    out = imai_sensitivity_rho(*_simple())
    assert out["acme_0"] == pytest.approx(1.2, abs=0.1)  # 0.8 * 1.5
    assert np.all(np.diff(out["acme"]) < 0)  # decreasing in rho


def test_sensIM_edge():
    x, m, y = _simple()
    at_crit = imai_sensitivity_rho(x, m, y, rho_grid=[imai_sensitivity_rho(x, m, y)["rho_critical"]])
    assert at_crit["acme"][0] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        imai_sensitivity_rho(x, m, y, rho_grid=[1.0])
