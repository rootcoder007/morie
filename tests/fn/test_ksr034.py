"""Tests for ksr034 (Kosorok shelf)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.ksr034 import kosorok_ch2_glivenko_cantelli_bracketing


def test_ksr034_basic():
    rng = np.random.default_rng(9)
    X = rng.random(150)
    F = [(lambda x, c=c: (np.asarray(x) <= c).astype(float)) for c in np.linspace(0.05, 0.95, 20)]
    out = kosorok_ch2_glivenko_cantelli_bracketing(F, X)
    assert out["finite_on_grid"] is True  # 'on grid', not 'for all eps'


def test_ksr034_edge():
    rng = np.random.default_rng(9)
    F = [(lambda x: (np.asarray(x) <= 0.5).astype(float))]
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_bracketing(F, rng.random(50), eps_grid=[0.0])
