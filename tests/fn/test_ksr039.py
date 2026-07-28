"""Tests for ksr039 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr039 import kosorok_ch2_weak_convergence_lipschitz


def test_ksr039_basic():
    rng = np.random.default_rng(10)
    A = rng.standard_normal(2000)
    same = kosorok_ch2_weak_convergence_lipschitz(A, rng.standard_normal(2000), rng=rng)
    shifted = kosorok_ch2_weak_convergence_lipschitz(A, rng.standard_normal(2000) + 2,
                                                     rng=rng)
    assert shifted["bl_distance"] > same["bl_distance"] * 3


def test_ksr039_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_lipschitz([1.0], [2.0])
