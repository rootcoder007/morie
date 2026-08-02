"""Tests for ksr034 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr034 import kosorok_ch2_glivenko_cantelli_bracketing


def test_ksr034_basic():
    rng = np.random.default_rng(9)
    X = rng.random(150)
    F = [(lambda x, c=c: (np.asarray(x) <= c).astype(float)) for c in np.linspace(.05,.95,20)]
    out = kosorok_ch2_glivenko_cantelli_bracketing(F, X)
    assert out["finite_on_grid"] is True  # 'on grid', not 'for all eps'


def test_ksr034_edge():
    rng = np.random.default_rng(9)
    F = [(lambda x: (np.asarray(x) <= 0.5).astype(float))]
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_bracketing(F, rng.random(50), eps_grid=[0.0])
