"""Tests for copfr."""

from morie.fn import _array_core as np
import pytest

from morie.fn._copula import copula_tau
from morie.fn.copfr import copula_frailty

def _pairs(seed=42, n=200, theta=2.0):
    rng = np.random.default_rng(seed)
    u = rng.random(n); w = rng.random(n)
    v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    t1 = -np.log(1 - u); t2 = -np.log(1 - v)
    c1 = rng.exponential(4.0, n); c2 = rng.exponential(4.0, n)
    return np.minimum(t1, c1), (t1 <= c1).astype(float), np.minimum(t2, c2), (t2 <= c2).astype(float)


def test_copfr_basic():
    t1, e1, t2, e2 = _pairs()
    out = copula_frailty(t1, e1, t2, e2, family="gumbel")
    assert copula_tau("gumbel", out["theta"]) == pytest.approx(out["tau_sample"], abs=1e-6)


def test_copfr_edge():
    t1, e1, t2, e2 = _pairs()
    ind = copula_frailty(t1, e1, t2, e2, family="independence")
    assert ind["joint_survival"] == pytest.approx(ind["s1"] * ind["s2"])
    with pytest.raises(ValueError):
        copula_frailty(t1, e1, t2, e2, family="weibull")
