"""Tests for ksr032 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr032 import kosorok_ch2_weak_convergence_iff


def test_ksr032_basic():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((300, 30))
    out = kosorok_ch2_weak_convergence_iff(rng.standard_normal((300, 30)), ref, eps=1.5)
    assert out["fidi_converged"] is True
    assert out["mean_gap"] < out["mean_tol"]  # tolerance is MC-scaled


def test_ksr032_edge():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((300, 30))
    with pytest.raises(ValueError):
        kosorok_ch2_weak_convergence_iff(ref, ref[:, :5])  # grid mismatch
