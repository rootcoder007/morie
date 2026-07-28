"""Tests for ksr056 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr056 import kosorok_ch2_lad_lipschitz_bound


def test_ksr056_basic():
    rng = np.random.default_rng(15)
    U = rng.standard_normal((100, 2))
    out = kosorok_ch2_lad_lipschitz_bound([1.0, 0.0], [0.8, 0.3], U,
                                          rng.standard_normal(100))
    assert out["bound_holds"] is True


def test_ksr056_edge():
    rng = np.random.default_rng(15)
    with pytest.raises(ValueError):
        kosorok_ch2_lad_lipschitz_bound([1.0], [1.0, 2.0],
                                        rng.standard_normal((10, 2)))
