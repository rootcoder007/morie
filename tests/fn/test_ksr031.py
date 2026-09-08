"""Tests for ksr031 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr031 import kosorok_ch2_weak_convergence_tightness


def test_ksr031_basic():
    rng = np.random.default_rng(6)
    grid = np.linspace(0, 1, 40)
    smooth = np.array([np.sin(2 * np.pi * grid + rng.random() * 6) for _ in range(150)])
    assert kosorok_ch2_weak_convergence_tightness(smooth, grid, eps=0.3)["decreasing"]


def test_ksr031_edge():
    rng = np.random.default_rng(6)
    P = rng.standard_normal((50, 20))
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_tightness(P, eps=0.0)
