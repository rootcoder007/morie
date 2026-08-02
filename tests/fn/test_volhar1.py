"""Tests for volhar1."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volhar import vol_har_rv
from morie.fn.volhar1 import vol_har_q


def test_volhar1_basic():
    rng = np.random.default_rng(42)
    rv = np.abs(rng.normal(1.0, 0.2, size=200)) + 0.1
    rq = rv**2 * (1 + rng.random(200))
    out = vol_har_q(rv, rq)
    assert out["coefficients"].size == 5
    assert out["r2"] >= vol_har_rv(rv)["r2"] - 1e-10  # nested


def test_volhar1_edge():
    with pytest.raises(ValueError):
        vol_har_q(np.ones(50), np.ones(40))
