"""Tests for volbpv."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volbpv import vol_bipower_variation


def test_volbpv_basic():
    r = np.array([0.1, -0.1, 0.1, -0.1])
    assert vol_bipower_variation(r)["bpv"] == pytest.approx(np.pi / 2 * 3 * 0.01)


def test_volbpv_edge():
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.01, size=300)
    clean = vol_bipower_variation(r)["bpv"]
    rj = r.copy(); rj[100] += 0.3
    assert vol_bipower_variation(rj)["bpv"] - clean < 0.02  # jump-robust
    with pytest.raises(ValueError):
        vol_bipower_variation([0.1, 0.2])
