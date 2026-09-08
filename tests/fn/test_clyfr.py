"""Tests for clyfr."""

from morie.fn import _array_core as np
import pytest

from morie.fn.clyfr import clayton_copula_frailty

def _pairs(seed=42, n=200, theta=2.0):
    rng = np.random.default_rng(seed)
    u = rng.random(n); w = rng.random(n)
    v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    t1 = -np.log(1 - u); t2 = -np.log(1 - v)
    c1 = rng.exponential(4.0, n); c2 = rng.exponential(4.0, n)
    return np.minimum(t1, c1), (t1 <= c1).astype(float), np.minimum(t2, c2), (t2 <= c2).astype(float)


def test_clyfr_basic():
    t1, e1, t2, e2 = _pairs()
    out = clayton_copula_frailty(t1, e1, t2, e2)
    assert out["tau"] == pytest.approx(0.5, abs=0.2)
    assert np.all(out["joint_survival"] <= np.minimum(out["s1"], out["s2"]) + 1e-8)


def test_clyfr_edge():
    t1, e1, t2, e2 = _pairs()
    with pytest.raises(ValueError):
        clayton_copula_frailty(t1, e1, t2, e2, theta=-1.0)
    with pytest.raises(ValueError):
        clayton_copula_frailty(t1[:3], e1[:3], t2[:3], e2[:3])
