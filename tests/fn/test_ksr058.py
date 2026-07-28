"""Tests for ksr058 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr058 import kosorok_ch2_law_iterated_logarithm


def test_ksr058_basic():
    rng = np.random.default_rng(16)
    out = kosorok_ch2_law_iterated_logarithm(rng.random(3000))
    assert out["lil_bound"] == 0.5  # eq. (2.21)
    assert 0 < out["lil_ratio"] < 0.5


def test_ksr058_edge():
    assert kosorok_ch2_law_iterated_logarithm(n=500)["chung_liminf_constant"] == \
        pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        kosorok_ch2_law_iterated_logarithm(np.random.default_rng(16).random(4))
