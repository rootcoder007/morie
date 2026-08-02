"""Tests for ksr041 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr041 import kosorok_ch2_bootstrap_donsker_almost_sure


def test_ksr041_basic():
    rng = np.random.default_rng(12)
    out = kosorok_ch2_bootstrap_donsker_almost_sure(rng.random(300), n_boot=200, rng=rng)
    assert out["both_conditions_met"] is True


def test_ksr041_edge():
    rng = np.random.default_rng(12)
    # a.s. convergence needs the extra envelope condition
    bad = kosorok_ch2_bootstrap_donsker_almost_sure(rng.random(300), n_boot=200,
                                                    rng=rng, envelope_sq_mean=np.inf)
    assert bad["both_conditions_met"] is False
