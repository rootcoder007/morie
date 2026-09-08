"""Tests for ksr068 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr068 import kosorok_ch3_cox_profile_score


def test_ksr068_basic():
    rng = np.random.default_rng(20)
    Z = rng.standard_normal((300, 1))
    T = rng.exponential(1.0 / np.exp(Z[:, 0] * 0.8))
    C = rng.exponential(2.0, 300)
    out = kosorok_ch3_cox_profile_score(Z=Z, time=np.minimum(T, C),
                                        event=(T <= C).astype(float))
    assert np.abs(out["score_at_root"]).max() < 1e-6


def test_ksr068_edge():
    rng = np.random.default_rng(20)
    with pytest.raises(ValueError):
        kosorok_ch3_cox_profile_score(Z=rng.standard_normal((50, 1)), time=None,
                                      event=np.ones(50))
