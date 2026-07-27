"""Tests for volrk."""

import numpy as np
import pytest

from morie.fn.volrk import vol_realised_kernel


def test_volrk_basic():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.01, size=400)
    out = vol_realised_kernel(r)
    assert out["rk"] == pytest.approx(0.01**2 * 400, rel=0.3)
    assert out["gammas"].size == out["H"] + 1


def test_volrk_edge():
    with pytest.raises(ValueError):
        vol_realised_kernel(np.ones(3))
    with pytest.raises(ValueError):
        vol_realised_kernel(np.ones(20), H=25)
