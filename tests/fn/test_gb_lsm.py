"""Tests for gb_lsm (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_lsm import gibbons_large_sample_moments


def test_gb_lsm_basic():
    rng = np.random.default_rng(3)
    approx = gibbons_large_sample_moments(15, 29)
    sims = np.sort(rng.standard_normal((4000, 29)), axis=1)[:, 14]
    assert approx["var"] == pytest.approx(sims.var(), rel=0.2)


def test_gb_lsm_edge():
    with pytest.raises(ValueError):
        gibbons_large_sample_moments(0, 10)
