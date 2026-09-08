"""Tests for evchibd.evt_chibar_dependence."""
from morie.fn import _array_core as np
from morie.fn.evchibd import evt_chibar_dependence


def test_perfect_dependence_chibar_high():
    x = [float(v) for v in np.random.default_rng(4).normal(0, 1, 800)._flat()]
    r = evt_chibar_dependence(x, x)
    # asymptotic dependence: chibar -> 1 (Coles p. 164 property 2)
    assert r["chibar_curve"][-1] > 0.8
    assert all(-1.0 <= v <= 1.0 for v in r["chibar_curve"])
