"""Tests for ksr040 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr040 import kosorok_ch2_bootstrap_donsker_iff


def test_ksr040_basic():
    rng = np.random.default_rng(11)
    out = kosorok_ch2_bootstrap_donsker_iff(rng.random(300), n_boot=400, rng=rng)
    assert out["max_abs_gap"] < 0.1  # matches the bridge covariance


def test_ksr040_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_bootstrap_donsker_iff(np.random.default_rng(11).random(5))
