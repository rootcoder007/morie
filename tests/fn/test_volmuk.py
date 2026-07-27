"""Tests for volmuk."""

import numpy as np
import pytest

from morie.fn.volmuk import vol_multi_kernel_rk


def test_volmuk_basic():
    rng = np.random.default_rng(42)
    r = rng.normal(scale=0.01, size=1200)
    out = vol_multi_kernel_rk(r, n_grids=3)
    assert out["rk_per_grid"].size == 3
    # measured 0.111 vs true 0.12 at m = 1200 (subgrids of 400)
    assert out["rk_avg"] == pytest.approx(0.01**2 * 1200, rel=0.35)


def test_volmuk_edge():
    with pytest.raises(ValueError):
        vol_multi_kernel_rk(np.ones(8), n_grids=3)
