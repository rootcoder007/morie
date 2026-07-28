"""Tests for ksr028 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr028 import kosorok_ch2_glivenko_cantelli_classical


def test_ksr028_basic():
    rng = np.random.default_rng(4)
    out = kosorok_ch2_glivenko_cantelli_classical(rng.random(3000))
    assert out["sup_distance"][-1] < out["sup_distance"][0]


def test_ksr028_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_classical([0.1, 0.2])
