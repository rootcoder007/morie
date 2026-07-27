"""Tests for volharj."""

import numpy as np
import pytest

from morie.fn.volharj import vol_har_rv_jump


def test_volharj_basic():
    rng = np.random.default_rng(42)
    rv = np.abs(rng.normal(1.0, 0.2, size=200)) + 0.1
    bpv = rv * (0.8 + 0.2 * rng.random(200))
    out = vol_har_rv_jump(rv, bpv)
    assert out["coefficients"].size == 5
    assert np.all(out["jump"] >= 0)


def test_volharj_edge():
    with pytest.raises(ValueError):
        vol_har_rv_jump(np.ones(50), -np.ones(50))
