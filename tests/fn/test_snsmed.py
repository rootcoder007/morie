"""Tests for snsmed.sensitivity_mediation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.snsmed import sensitivity_mediation


def _simple(seed=42, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    return x, m, y


def test_snsmed_basic():
    out = sensitivity_mediation(*_simple(), rho=[0.0, 0.3])
    assert out["acme"][0] == pytest.approx(1.2, abs=0.1)
    assert out["acme"][1] < out["acme"][0]


def test_snsmed_edge():
    with pytest.raises(ValueError):
        sensitivity_mediation(*_simple(), rho=[-1.0])
